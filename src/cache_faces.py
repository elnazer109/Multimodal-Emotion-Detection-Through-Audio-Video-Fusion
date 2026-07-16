"""
Pre-extract RAVDESS face crops once, as uint8, so training runs never pay for it again.

Two deliberate departures from the original notebook's `extract_video_frames`:

  1. Stores uint8 (T,H,W,C) instead of float32 (C,T,H,W) with per-clip standardization
     already applied. The original bakes the normalization into the cache, so changing
     it means re-extracting all 2452 clips. Normalization belongs at load time.
     Also 4x smaller: ~1.5GB vs ~5.9GB.

  2. Pins video modality to 01 (full-AV). The original's pair key omits `modality`, so
     each key matches both the 01 and 02 copies and `.iloc[0]` picks by filesystem glob
     order. Same pixels either way -- this is for determinism, not accuracy.

CPU-only by design: Haar cascade is CPU-bound, so burning GPU quota here would be waste.
"""
import os, re, glob, json
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
from tqdm import tqdm

N_FRAMES = 16
IMG_SIZE = 112
OUT_DIR = "/kaggle/working"

EMOTION_MAP = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
               5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
EMOTIONS = list(EMOTION_MAP.values())
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def find_root():
    for base in glob.glob("/kaggle/input/**", recursive=True):
        if os.path.basename(base) == "Audio_Song_Actors_01-24":
            return os.path.dirname(base)
    raise FileNotFoundError("Could not locate RAVDESS root under /kaggle/input")


def parse(path):
    base = os.path.splitext(os.path.basename(path))[0]
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = \
        [int(p) for p in base.split("-")]
    return dict(
        path=path, modality=modality, vocal_channel=vocal_channel,
        emotion=emotion, emotion_label=EMOTION_MAP[emotion],
        emotion_idx=EMOTION_TO_IDX[EMOTION_MAP[emotion]],
        intensity=intensity, statement=statement, repetition=repetition, actor=actor,
        # modality included so the 01/02 video copies no longer collide
        key=f"{vocal_channel:02d}-{emotion:02d}-{intensity:02d}-"
            f"{statement:02d}-{repetition:02d}-{actor:02d}",
    )


_cascade = None


def _crop_face(frame):
    global _cascade
    if _cascade is None:  # built per-worker; the classifier is not fork-safe to share
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return frame[y:y + h, x:x + w], True


def extract(path):
    """Returns (T,H,W,C) uint8 and the number of frames where a face was actually found."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int).tolist())
    frames, hits = [], 0
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            face, found = _crop_face(frame)
            hits += found
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    cap.release()
    while len(frames) < N_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8))
    return np.stack(frames[:N_FRAMES]).astype(np.uint8), hits


def work(args):
    i, path = args
    try:
        arr, hits = extract(path)
        return i, arr, hits, ""
    except Exception as e:
        return i, np.zeros((N_FRAMES, IMG_SIZE, IMG_SIZE, 3), np.uint8), 0, str(e)


if __name__ == "__main__":
    root = find_root()
    print("ROOT:", root, flush=True)

    audio = [p for d in glob.glob(os.path.join(root, "**", "Audio_*Actors_01-24"), recursive=True)
             for p in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)]
    video = [p for d in glob.glob(os.path.join(root, "**", "Video_*Actor_*"), recursive=True)
             for p in glob.glob(os.path.join(d, "**", "*.mp4"), recursive=True)]
    print(f"found {len(audio)} audio, {len(video)} video", flush=True)

    adf = pd.DataFrame([parse(p) for p in audio])
    vdf = pd.DataFrame([parse(p) for p in video])

    # The collision, made explicit rather than resolved by glob order.
    dupes = vdf.groupby("key").size()
    print(f"video keys matching >1 file: {(dupes > 1).sum()} / {len(dupes)}", flush=True)
    vdf = vdf[vdf.modality == 1].copy()
    assert vdf.groupby("key").size().max() == 1, "still colliding after pinning modality=01"

    pairs = (adf.merge(vdf, on="key", suffixes=("_a", "_v"))
                .rename(columns={"path_a": "audio_path", "path_v": "video_path",
                                 "emotion_idx_a": "emotion_idx", "emotion_label_a": "emotion_label",
                                 "actor_a": "actor"})
                .sort_values("key").reset_index(drop=True))
    print(f"paired clips: {len(pairs)}", flush=True)
    assert len(pairs) == 2452, f"expected 2452 pairs, got {len(pairs)}"

    # Sanity-check the speech/song split the notebook's cell-5 comment gets backwards.
    print(pairs.groupby("vocal_channel_a").size().to_dict(), flush=True)
    print(pairs["emotion_label"].value_counts().to_dict(), flush=True)

    out = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, "faces_u8.npy"), mode="w+",
        dtype=np.uint8, shape=(len(pairs), N_FRAMES, IMG_SIZE, IMG_SIZE, 3))

    face_hits, errors = np.zeros(len(pairs), np.int16), {}
    with Pool(4) as pool:
        for i, arr, hits, err in tqdm(
                pool.imap_unordered(work, list(enumerate(pairs["video_path"])), chunksize=8),
                total=len(pairs), desc="extracting"):
            out[i], face_hits[i] = arr, hits
            if err:
                errors[int(i)] = err
    out.flush()

    pairs["face_hits"] = face_hits
    pairs.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)

    # A clip with 0/16 detections is 16 full frames, not 16 faces -- worth knowing before
    # anyone reads the accuracy number.
    print(f"\nclips with zero face detections: {(face_hits == 0).sum()} / {len(pairs)}", flush=True)
    print(f"mean faces found per clip: {face_hits.mean():.2f} / {N_FRAMES}", flush=True)
    print(f"decode errors: {len(errors)}", flush=True)
    if errors:
        print(json.dumps(dict(list(errors.items())[:5]), indent=2), flush=True)
    print(f"\ncache: {out.nbytes / 1e9:.2f} GB", flush=True)
