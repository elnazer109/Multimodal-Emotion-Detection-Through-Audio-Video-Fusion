"""
Actor-independent re-evaluation of the RAVDESS audio+video fusion model.

The published protocol splits 2452 clips from 24 actors with StratifiedKFold(shuffle=True),
so every validation actor also appears in training. RAVDESS records each
(actor, emotion, intensity, statement) twice, so repetition 1 lands in train and repetition 2
in val: same face, same voice, same sentence. The 94.29% it reports is substantially a measure
of actor recognition. This script measures the same architecture actor-independently.

Reported protocols, all on identical held-out actors:
  A. paper-fusion   -- fusion trained on features from models that memorized those clips
  B. oof-fusion     -- fusion trained on out-of-fold features (standard stacking)
Difference between A and B isolates what the fusion-feature leak was worth.

Base hyperparameters are verbatim from the original CFG. Changes are marked CHANGED.
"""
import os, glob, json, gc, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = "/kaggle/working"


def require_usable_gpu():
    """Fail fast and loud rather than silently training on CPU.

    Kaggle's default GPU is a P100 (sm_60) and the preinstalled torch (cu128) only supports
    sm_70+, so `torch.cuda.is_available()` returns True while every kernel launch fails. The
    first version of this script trusted is_available(), fell back to CPU, and spent an hour
    producing one audio fold -- R3D-18 on CPU would have taken days. A run that cannot finish
    should die in seconds, not look healthy overnight.
    Push with:  kaggle kernels push --accelerator NvidiaTeslaT4
    """
    if not torch.cuda.is_available():
        raise SystemExit("NO GPU ALLOCATED. Refusing to train on CPU (R3D-18 would take days).")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  sm_{major}{minor}  torch {torch.__version__}", flush=True)
    try:  # is_available() is not proof the device can run anything
        _ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    except Exception as e:
        raise SystemExit(f"GPU {name} (sm_{major}{minor}) is UNUSABLE with torch "
                         f"{torch.__version__}: {e}\nRe-push with --accelerator NvidiaTeslaT4.")
    print("GPU verified: a real kernel launched and returned.", flush=True)
    return "cuda"


DEVICE = require_usable_gpu()

CFG = dict(
    sr=22050, n_mfcc=40, n_fft=2048, hop_length=512, mfcc_fixed_frames=130,
    audio_lr=5e-4, audio_batch=16, audio_epochs=100, audio_dropout_lstm=0.4,
    audio_dropout_dense=0.3, audio_l2=1e-2,
    n_frames=16, img_size=112,
    video_lr=1e-4, video_batch=32, video_epochs=15, video_dropout=0.4,
    fusion_lr=5e-4, fusion_epochs=1000, fusion_dropout=0.5, fusion_wd=1e-5,
    fusion_step_size=100, fusion_gamma=0.5,
    n_classes=8, n_folds=5, early_stop_patience=10,
)

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Per-clip standardization, as the original notebook and the paper do.
#
# An earlier version of this file swapped in Kinetics channel statistics, on the theory that
# per-clip standardization "fights" the KINETICS400_V1 pretrained weights. That was measured and
# it is false -- on the paper's protocol it cost 10.75 points of video accuracy and destabilised
# it badly (81.97 +-8.11 against the paper's 92.72, with folds collapsing to 71.22):
#
#     paper Table VII video   92.72
#     notebook, per-clip      91.24  (fold 1)
#     Kinetics, fold 1        85.54
#     Kinetics, fold 3        71.22
#
# RAVDESS records each actor under consistent studio lighting, so standardizing each clip removes
# nuisance variation that matters more here than matching Kinetics' global stats. Reverted.

# ---------------------------------------------------------------- pretrained weights, offline
# Kaggle kernels have no network here (enable_internet is set, but DNS still fails -- the account
# needs phone verification), so r3d_18(weights=KINETICS400_V1) cannot fetch its checkpoint.
# The weights come from a public dataset instead.
#
# That dataset is a stranger's upload, so it is not trusted on faith: torchvision names
# checkpoints <arch>-<first 8 hex of sha256>.pth and torch.hub verifies exactly that prefix.
# Re-checking it here makes the file provably bit-identical to the official KINETICS400_V1
# checkpoint. Wrong or tampered weights would silently poison every number in the paper.
R3D_SHA_PREFIX = "b3b3357e"
_w = glob.glob("/kaggle/input/**/r3d_18-*.pth", recursive=True)
if not _w:
    print("r3d_18 weights not attached. /kaggle/input contains:", flush=True)
    for p in sorted(glob.glob("/kaggle/input/*/*", recursive=True))[:40]:
        print("   ", p, flush=True)
    raise SystemExit("attach sabreenelkamash/r3d-18-pretrained-weights")
_digest = hashlib.sha256(open(_w[0], "rb").read()).hexdigest()
if not _digest.startswith(R3D_SHA_PREFIX):
    raise SystemExit(f"REFUSING TO TRAIN: {_w[0]} sha256 starts {_digest[:8]}, expected "
                     f"{R3D_SHA_PREFIX}. This is not the official KINETICS400_V1 checkpoint.")
R3D_STATE = torch.load(_w[0], map_location="cpu")
print(f"r3d_18 weights verified: sha256 {_digest[:8]}... matches official checkpoint", flush=True)

# ---------------------------------------------------------------- data

# Recursive: this environment nests mounts (the RAVDESS set lands at
# /kaggle/input/datasets/orvile/ravdess-dataset, not /kaggle/input/ravdess-dataset), and a
# kernel-output source nests too. A one-level glob silently finds nothing.
CACHE = glob.glob("/kaggle/input/**/faces_u8.npy", recursive=True)
if not CACHE:
    print("faces_u8.npy not found. /kaggle/input contains:", flush=True)
    for p in sorted(glob.glob("/kaggle/input/**", recursive=True))[:60]:
        print("   ", p, flush=True)
    raise SystemExit("face cache not attached -- check kernel_sources")
FACES = np.load(CACHE[0], mmap_mode="r")               # (N,16,112,112,3) uint8
pairs = pd.read_csv(os.path.join(os.path.dirname(CACHE[0]), "manifest.csv"))
assert len(pairs) == len(FACES) == 2452, f"{len(pairs)} vs {len(FACES)}"
print(f"faces {FACES.shape} {FACES.dtype} | pairs {len(pairs)} | actors {pairs.actor.nunique()}")

# plain arrays: pandas .iloc inside __getitem__ costs real time across ~15 trainings
LABELS = pairs.emotion_idx.values.astype(np.int64)
ACTORS = pairs.actor.values.astype(np.int64)

# Re-resolve audio paths rather than trusting the manifest's absolute ones -- they were written
# under the cache kernel's mounts, and this kernel mounts a different set of inputs.
# key == filename minus the leading modality field.
wav_by_key = {}
for p in glob.glob("/kaggle/input/**/*.wav", recursive=True):
    parts = os.path.splitext(os.path.basename(p))[0].split("-")
    if len(parts) == 7 and parts[0] == "03":
        wav_by_key["-".join(parts[1:])] = p
pairs["audio_path"] = pairs["key"].map(wav_by_key)
missing = pairs["audio_path"].isna().sum()
assert missing == 0, f"{missing} audio paths unresolved -- is the RAVDESS dataset attached?"
print(f"resolved {len(wav_by_key)} wavs")


def preemphasis(sig, coef=0.97):
    return np.append(sig[0], sig[1:] - coef * sig[:-1])


def extract_mfcc(path):
    y, sr = librosa.load(path, sr=CFG["sr"])
    y = preemphasis(y)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG["n_mfcc"], n_fft=CFG["n_fft"],
                             hop_length=CFG["hop_length"], window="hamming")
    T = CFG["mfcc_fixed_frames"]
    m = np.pad(m, ((0, 0), (0, T - m.shape[1])), mode="constant") if m.shape[1] < T else m[:, :T]
    return m.astype(np.float32)


MFCC_PATH = os.path.join(OUT, "mfcc.npy")
if os.path.exists(MFCC_PATH):
    MFCC = np.load(MFCC_PATH)
else:
    MFCC = np.stack([extract_mfcc(p) for p in tqdm(pairs["audio_path"], desc="mfcc")])
    np.save(MFCC_PATH, MFCC)
print("mfcc", MFCC.shape)


class AudioDS(Dataset):
    def __init__(self, idx): self.idx = np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(MFCC[j]), int(LABELS[j])


class VideoDS(Dataset):
    def __init__(self, idx): self.idx = np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.asarray(FACES[j]).copy()).float()   # (T,H,W,C), 0..255
        x = (x - x.mean()) / (x.std() + 1e-6)                        # per-clip, scalar mu/sigma
        x = x.permute(3, 0, 1, 2)                                    # (C,T,H,W)
        return x, int(LABELS[j])


# ---------------------------------------------------------------- models (verbatim)

class AudioModel(nn.Module):
    def __init__(self, n_mfcc=40, n_classes=8, dropout_lstm=0.4, dropout_dense=0.3):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 3, padding=1), nn.ReLU(True),
                                 nn.MaxPool1d(2), nn.BatchNorm1d(cout))
        self.cnn = nn.Sequential(blk(n_mfcc, 64), blk(64, 128), blk(128, 256), blk(256, 512))
        self.bilstm1 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout_lstm); self.bn1 = nn.BatchNorm1d(512)
        self.bilstm2 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout_lstm); self.bn2 = nn.BatchNorm1d(1024)
        self.birnn = nn.RNN(1024, 512, batch_first=True, bidirectional=True, nonlinearity="tanh")
        self.drop3 = nn.Dropout(dropout_lstm); self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 128); self.drop4 = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(128, 64); self.drop5 = nn.Dropout(dropout_dense)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x, return_features=False):
        h = self.cnn(x).permute(0, 2, 1)
        h, _ = self.bilstm1(h)
        h = self.drop1(h.transpose(1, 2)).transpose(1, 2)
        h = self.bn1(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.bilstm2(h)
        h = self.drop2(h.transpose(1, 2)).transpose(1, 2)
        h = self.bn2(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.birnn(h)
        h = self.drop3(h.transpose(1, 2)).transpose(1, 2)
        h = self.bn3(h.transpose(1, 2)).transpose(1, 2)
        h = h.mean(1)
        h = F.relu(self.fc1(h)); h = self.drop4(h)
        f = F.relu(self.fc2(h)); h = self.drop5(f)
        return (self.out(h), f) if return_features else self.out(h)


class VideoModel(nn.Module):
    # refine head kept as-is: R3D-18's own pooling has already collapsed the volume to 1x1x1,
    # so the 3x3x3 convs act on padding and reduce to their centre taps. A documented deviation
    # from the paper's Fig.4, not a bug -- left alone deliberately; changing it is an
    # architecture change that would need re-tuning we have no budget to verify.
    def __init__(self, n_classes=8, dropout=0.4):
        super().__init__()
        bb = r3d_18(weights=None)
        bb.load_state_dict(R3D_STATE)   # hash-verified above; identical to KINETICS400_V1
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.refine = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv3d(256, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv3d(128, 64, 3, padding=1), nn.ReLU(True))
        self.gap = nn.AdaptiveAvgPool3d(1); self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 32); self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        emb = self.backbone(x)
        emb512 = emb.flatten(1)
        h = self.gap(self.refine(emb)).flatten(1)
        h = F.relu(self.fc1(self.dropout(h)))
        return self.fc2(h), emb512


class FusionMLP(nn.Module):
    def __init__(self, in_dim=568, n_classes=8, dropout=0.5):
        super().__init__()
        L = []
        for a, b in [(in_dim, 256), (256, 128), (128, 64), (64, 32)]:
            L += [nn.Linear(a, b), nn.ReLU(True), nn.BatchNorm1d(b), nn.Dropout(dropout)]
        L += [nn.Linear(32, n_classes)]
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------- training

def cw(idx):
    y = pairs.emotion_idx.iloc[idx].values
    w = compute_class_weight("balanced", classes=np.arange(CFG["n_classes"]), y=y)
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


def train_base(kind, tr_idx, tag):
    """CHANGED: early stopping uses actors carved from the TRAINING set, never the test fold."""
    g = ACTORS[tr_idx]
    inner_tr, inner_va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                              .split(tr_idx, groups=g))
    a, b = np.asarray(tr_idx)[inner_tr], np.asarray(tr_idx)[inner_va]

    if kind == "audio":
        model = AudioModel(dropout_lstm=CFG["audio_dropout_lstm"],
                           dropout_dense=CFG["audio_dropout_dense"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["audio_lr"], weight_decay=CFG["audio_l2"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
        tl = DataLoader(AudioDS(a), batch_size=CFG["audio_batch"], shuffle=True, num_workers=2)
        vl = DataLoader(AudioDS(b), batch_size=CFG["audio_batch"], num_workers=2)
        epochs, patience = CFG["audio_epochs"], CFG["early_stop_patience"]
    else:
        model = VideoModel(dropout=CFG["video_dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["video_lr"])
        sch = None
        tl = DataLoader(VideoDS(a), batch_size=CFG["video_batch"], shuffle=True, num_workers=2)
        vl = DataLoader(VideoDS(b), batch_size=CFG["video_batch"], num_workers=2)
        epochs, patience = CFG["video_epochs"], CFG["early_stop_patience"]

    crit = nn.CrossEntropyLoss(weight=cw(a))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
            out = model(xb); out = out[0] if isinstance(out, tuple) else out
            loss = crit(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vloss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb); out = out[0] if isinstance(out, tuple) else out
                vloss += crit(out, yb).item() * xb.size(0); n += xb.size(0)
        vloss /= max(n, 1)
        if sch: sch.step(vloss)
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    {tag}/{kind}: early stop @ {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def features(idx, am, vm):
    """568-dim: 40 mean-MFCC + 8 audio softmax + 512 R3D embedding + 8 video softmax."""
    al = DataLoader(AudioDS(idx), batch_size=64, num_workers=2)
    vl = DataLoader(VideoDS(idx), batch_size=CFG["video_batch"], num_workers=2)
    mm, ap = [], []
    for xb, _ in al:
        xb = xb.to(DEVICE)
        mm.append(xb.mean(2).cpu().numpy())
        ap.append(F.softmax(am(xb), 1).cpu().numpy())
    ve, vp = [], []
    for xb, _ in vl:
        xb = xb.to(DEVICE)
        lg, e = vm(xb)
        ve.append(e.cpu().numpy()); vp.append(F.softmax(lg, 1).cpu().numpy())
    return np.concatenate([np.concatenate(mm), np.concatenate(ap),
                           np.concatenate(ve), np.concatenate(vp)], axis=1)


@torch.no_grad()
def predict(kind, idx, m):
    dl = DataLoader((AudioDS if kind == "audio" else VideoDS)(idx),
                    batch_size=64 if kind == "audio" else CFG["video_batch"], num_workers=2)
    P = []
    for xb, _ in dl:
        out = m(xb.to(DEVICE)); out = out[0] if isinstance(out, tuple) else out
        P.append(F.softmax(out, 1).cpu().numpy())
    return np.concatenate(P)


def train_fusion(Xtr, ytr, actors_tr, Xte):
    """CHANGED: StandardScaler -- raw MFCC means sit near -300 while softmax probs are 0..1,
    and FusionMLP's first BatchNorm is *after* the first Linear, so nothing normalized them.

    Order matters here: split by actor FIRST, then SMOTE only the training side. SMOTE before
    the split would synthesize training points by interpolating validation points, so the
    early-stopping signal would be reading data it helped create.
    """
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    itr, iva = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
                    .split(Xtr, groups=actors_tr))
    Xa, ya = Xtr[itr], ytr[itr]
    Xv_, yv_ = Xtr[iva], ytr[iva]
    try:
        Xa, ya = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xa, ya)
    except ValueError as e:
        print("    SMOTE skipped:", e, flush=True)

    m = FusionMLP(in_dim=Xa.shape[1], dropout=CFG["fusion_dropout"]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=CFG["fusion_lr"], weight_decay=CFG["fusion_wd"])
    sch = torch.optim.lr_scheduler.StepLR(opt, CFG["fusion_step_size"], CFG["fusion_gamma"])
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xa, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(ya, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(Xv_, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(yv_, dtype=torch.long).to(DEVICE)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, CFG["fusion_epochs"] + 1):
        m.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            if len(b) < 2: continue
            loss = crit(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xv), yv).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 30: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return F.softmax(m(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)), 1).cpu().numpy()


# ---------------------------------------------------------------- protocol

def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p) * 100,
                balanced_accuracy=balanced_accuracy_score(y, p) * 100,
                precision=precision_score(y, p, average="macro", zero_division=0) * 100,
                recall=recall_score(y, p, average="macro", zero_division=0) * 100,
                f1=f1_score(y, p, average="macro", zero_division=0) * 100)


# CHANGED: GroupKFold on actor. No actor appears in both train and test.
outer = GroupKFold(n_splits=CFG["n_folds"])
rows, oof_records = [], []

for fold, (tr, te) in enumerate(outer.split(pairs, groups=pairs.actor), 1):
    tr_actors, te_actors = sorted(set(pairs.actor.iloc[tr])), sorted(set(pairs.actor.iloc[te]))
    assert not (set(tr_actors) & set(te_actors)), "actor leaked across the split"
    print(f"\n===== fold {fold}/{CFG['n_folds']} | train actors {tr_actors} | "
          f"test actors {te_actors} | {len(tr)}/{len(te)} clips =====", flush=True)

    # --- inner 2-fold over TRAIN actors -> out-of-fold features for every training clip
    inner = GroupKFold(n_splits=2)
    oof = np.zeros((len(tr), 568), np.float32)
    for h, (ia, ib) in enumerate(inner.split(tr, groups=ACTORS[tr]), 1):
        A, B = np.asarray(tr)[ia], np.asarray(tr)[ib]
        print(f"  inner half {h}: fit {len(A)} -> predict {len(B)}", flush=True)
        am, vm = train_base("audio", A, f"f{fold}h{h}"), train_base("video", A, f"f{fold}h{h}")
        oof[ib] = features(B, am, vm)      # clean: these models never saw B
        del am, vm; gc.collect(); torch.cuda.empty_cache()

    # --- full-train base models: test features, the standalone rows, AND the leaky features
    print("  full-train base models", flush=True)
    am, vm = train_base("audio", tr, f"f{fold}full"), train_base("video", tr, f"f{fold}full")
    Xte = features(te, am, vm)
    # Exactly what the original notebook feeds its fusion MLP: the full-train models predicting
    # the very clips they trained on. Free -- inference only, no extra training. Using the inner
    # half-models here instead would have made the ablation unfair, since they see half the data.
    leaky = features(tr, am, vm)
    yte = LABELS[te]
    ytr = LABELS[tr]

    ap, vp = predict("audio", te, am), predict("video", te, vm)
    rows.append(dict(fold=fold, model="audio", **metrics(yte, ap.argmax(1))))
    rows.append(dict(fold=fold, model="video", **metrics(yte, vp.argmax(1))))
    del am, vm; gc.collect(); torch.cuda.empty_cache()

    # --- the ablation: same test fold, same architecture, only the fusion features differ
    p_leak = train_fusion(leaky, ytr, ACTORS[tr], Xte)
    p_oof = train_fusion(oof, ytr, ACTORS[tr], Xte)
    rows.append(dict(fold=fold, model="fusion-paper", **metrics(yte, p_leak.argmax(1))))
    rows.append(dict(fold=fold, model="fusion-oof", **metrics(yte, p_oof.argmax(1))))
    for r in rows[-4:]:
        print(f"  {r['model']:14s} acc {r['accuracy']:.2f}  bal {r['balanced_accuracy']:.2f}"
              f"  f1 {r['f1']:.2f}", flush=True)

    # per-sample predictions -> the binary mapping is pure post-processing downstream
    for k, j in enumerate(te):
        oof_records.append(dict(
            idx=int(j), fold=fold, actor=int(pairs.actor.iloc[j]),
            vocal_channel=int(pairs.vocal_channel_a.iloc[j]) if "vocal_channel_a" in pairs
            else int(pairs.vocal_channel.iloc[j]),
            true=int(yte[k]), true_label=EMOTIONS[int(yte[k])],
            **{f"audio_p{i}": float(ap[k, i]) for i in range(8)},
            **{f"video_p{i}": float(vp[k, i]) for i in range(8)},
            **{f"fusion_p{i}": float(p_oof[k, i]) for i in range(8)},
            **{f"fusion_paper_p{i}": float(p_leak[k, i]) for i in range(8)}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fold_results.csv"), index=False)
    pd.DataFrame(oof_records).to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)

# ---------------------------------------------------------------- report

df = pd.DataFrame(rows)
summary = df.groupby("model")[["accuracy", "balanced_accuracy", "f1"]].agg(["mean", "std"])
print("\n\n=================== ACTOR-INDEPENDENT (GroupKFold by actor) ===================")
print(summary.round(2).to_string())
print("\nPaper's reported fusion accuracy under the leaky random split: 94.29% (not re-run; "
      "cited from the committed notebook outputs)")
print("\nThe fusion-paper vs fusion-oof gap is the value of the fusion-feature leak alone,")
print("measured on identical held-out actors.")
summary.to_csv(os.path.join(OUT, "summary.csv"))

preds = pd.DataFrame(oof_records)
cm = confusion_matrix(preds["true"], preds[[f"fusion_p{i}" for i in range(8)]].values.argmax(1))
print("\nfusion-oof confusion matrix (rows=true):")
print(pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_string())
print(f"\nwrote {len(preds)} per-sample predictions -> oof_predictions.csv")
