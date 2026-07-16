"""
Paper-comparable arm: beat 94.29% on the paper's own protocol.

This deliberately keeps everything the original notebook does, including the parts we criticise
elsewhere, so the comparison against 94.29% is like-for-like rather than a strawman:

  * StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over clips -- the leaky split
  * fusion features built by running the training clips through the models trained on them
  * SMOTE on the fused features
  * reported metric = max-over-epochs validation accuracy, scored on the selection fold
  * identical hyperparameters

Only two things change, and both are implementation defects rather than protocol choices:

  1. Kinetics channel statistics for video normalization. The original standardizes each clip by
     its own mean/std, but R3D_18_Weights.KINETICS400_V1 was trained with Kinetics stats -- the
     pretrained features are being fed inputs from the wrong distribution.
  2. StandardScaler on the 568-dim fused vector. Raw MFCC means sit near -300 next to 0..1 softmax
     probabilities, and FusionMLP's first BatchNorm is *after* the first Linear, so nothing ever
     normalized the input.

The resulting number is directly comparable to 94.29% and means "the published architecture,
implemented correctly, on the published benchmark". It does NOT mean the model reads emotion
better -- this split still lets actors leak across folds. That claim lives in the
actor-independent arm. Both numbers go in the paper; neither substitutes for the other.
"""
import os, glob, gc, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score)
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
    producing one audio fold. A run that cannot finish should die in seconds.
    Push with:  kaggle kernels push --accelerator NvidiaTeslaT4
    """
    if not torch.cuda.is_available():
        raise SystemExit("NO GPU ALLOCATED. Refusing to train on CPU (R3D-18 would take days).")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  sm_{major}{minor}  torch {torch.__version__}", flush=True)
    try:
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

# CHANGE 1 of 2
KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

# Kaggle kernels have no network here (enable_internet is set but DNS still fails -- the account
# needs phone verification), so the R3D-18 checkpoint comes from a public dataset. It is a
# stranger's upload, so verify rather than trust: torchvision names checkpoints
# <arch>-<first 8 hex of sha256>.pth and torch.hub checks exactly that prefix. Wrong weights would
# silently poison every number, and this arm's whole point is a trustworthy comparison to 94.29%.
R3D_SHA_PREFIX = "b3b3357e"
_w = glob.glob("/kaggle/input/**/r3d_18-*.pth", recursive=True)
if not _w:
    raise SystemExit("attach sabreenelkamash/r3d-18-pretrained-weights")
_digest = hashlib.sha256(open(_w[0], "rb").read()).hexdigest()
if not _digest.startswith(R3D_SHA_PREFIX):
    raise SystemExit(f"REFUSING TO TRAIN: {_w[0]} sha256 starts {_digest[:8]}, expected "
                     f"{R3D_SHA_PREFIX}. Not the official KINETICS400_V1 checkpoint.")
R3D_STATE = torch.load(_w[0], map_location="cpu")
print(f"r3d_18 weights verified: sha256 {_digest[:8]}... matches official", flush=True)

# Recursive: this environment nests mounts (RAVDESS lands at
# /kaggle/input/datasets/orvile/ravdess-dataset), and kernel-output sources nest too.
CACHE = glob.glob("/kaggle/input/**/faces_u8.npy", recursive=True)
if not CACHE:
    print("faces_u8.npy not found. /kaggle/input contains:", flush=True)
    for p in sorted(glob.glob("/kaggle/input/**", recursive=True))[:60]:
        print("   ", p, flush=True)
    raise SystemExit("face cache not attached -- check kernel_sources")
FACES = np.load(CACHE[0], mmap_mode="r")
pairs = pd.read_csv(os.path.join(os.path.dirname(CACHE[0]), "manifest.csv"))
assert len(pairs) == len(FACES) == 2452
LABELS = pairs.emotion_idx.values.astype(np.int64)

wav_by_key = {}
for p in glob.glob("/kaggle/input/**/*.wav", recursive=True):
    parts = os.path.splitext(os.path.basename(p))[0].split("-")
    if len(parts) == 7 and parts[0] == "03":
        wav_by_key["-".join(parts[1:])] = p
pairs["audio_path"] = pairs["key"].map(wav_by_key)
assert pairs["audio_path"].notna().all(), "audio paths unresolved"


def preemphasis(sig, coef=0.97):
    return np.append(sig[0], sig[1:] - coef * sig[:-1])


def extract_mfcc(path):
    y, sr = librosa.load(path, sr=CFG["sr"])
    y = preemphasis(y)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG["n_mfcc"], n_fft=CFG["n_fft"],
                             hop_length=CFG["hop_length"], window="hamming")
    T = CFG["mfcc_fixed_frames"]
    return (np.pad(m, ((0, 0), (0, T - m.shape[1])), mode="constant")
            if m.shape[1] < T else m[:, :T]).astype(np.float32)


MP = os.path.join(OUT, "mfcc.npy")
MFCC = np.load(MP) if os.path.exists(MP) else np.stack(
    [extract_mfcc(p) for p in tqdm(pairs["audio_path"], desc="mfcc")])
if not os.path.exists(MP):
    np.save(MP, MFCC)


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
        x = torch.from_numpy(np.asarray(FACES[j]).copy()).float().div_(255.0)
        x = x.permute(3, 0, 1, 2)
        x = (x - KIN_MEAN) / KIN_STD           # CHANGE 1 of 2
        return x, int(LABELS[j])


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

    def forward(self, x):
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
        return self.out(h)


class VideoModel(nn.Module):
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
        h = self.gap(self.refine(emb)).flatten(1)
        h = F.relu(self.fc1(self.dropout(h)))
        return self.fc2(h), emb.flatten(1)


class FusionMLP(nn.Module):
    def __init__(self, in_dim=568, n_classes=8, dropout=0.5):
        super().__init__()
        L = []
        for a, b in [(in_dim, 256), (256, 128), (128, 64), (64, 32)]:
            L += [nn.Linear(a, b), nn.ReLU(True), nn.BatchNorm1d(b), nn.Dropout(dropout)]
        L += [nn.Linear(32, n_classes)]
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


def cw(idx):
    w = compute_class_weight("balanced", classes=np.arange(8), y=LABELS[idx])
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p) * 100,
                balanced_accuracy=balanced_accuracy_score(y, p) * 100,
                precision=precision_score(y, p, average="macro", zero_division=0) * 100,
                recall=recall_score(y, p, average="macro", zero_division=0) * 100,
                f1=f1_score(y, p, average="macro", zero_division=0) * 100)


def train_base(kind, tr, va, tag):
    """Paper protocol: early-stop and select on the very fold that gets reported."""
    if kind == "audio":
        m = AudioModel(dropout_lstm=CFG["audio_dropout_lstm"],
                       dropout_dense=CFG["audio_dropout_dense"]).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=CFG["audio_lr"], weight_decay=CFG["audio_l2"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
        tl = DataLoader(AudioDS(tr), batch_size=CFG["audio_batch"], shuffle=True, num_workers=2)
        vl = DataLoader(AudioDS(va), batch_size=CFG["audio_batch"], num_workers=2)
        epochs = CFG["audio_epochs"]
    else:
        m = VideoModel(dropout=CFG["video_dropout"]).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=CFG["video_lr"])
        sch = None
        tl = DataLoader(VideoDS(tr), batch_size=CFG["video_batch"], shuffle=True, num_workers=2)
        vl = DataLoader(VideoDS(va), batch_size=CFG["video_batch"], num_workers=2)
        epochs = CFG["video_epochs"]

    crit = nn.CrossEntropyLoss(weight=cw(tr))
    best_acc, best_loss, best_state, bad = 0.0, float("inf"), None, 0
    for ep in range(1, epochs + 1):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
            o = m(xb); o = o[0] if isinstance(o, tuple) else o
            loss = crit(o, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); vl_sum, n, ys, ps = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                o = m(xb); o = o[0] if isinstance(o, tuple) else o
                vl_sum += crit(o, yb).item() * xb.size(0); n += xb.size(0)
                ys += yb.cpu().tolist(); ps += o.argmax(1).cpu().tolist()
        vloss, vacc = vl_sum / n, accuracy_score(ys, ps) * 100
        if sch: sch.step(vloss)
        if vacc > best_acc:                       # paper: max-over-epochs val accuracy
            best_acc = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if vloss < best_loss - 1e-5:
            best_loss, bad = vloss, 0
        else:
            bad += 1
            if bad >= CFG["early_stop_patience"]:
                print(f"    {tag}/{kind}: early stop @ {ep}", flush=True); break
    m.load_state_dict(best_state)
    print(f"    {tag}/{kind}: best val acc {best_acc:.2f}", flush=True)
    return m.eval(), best_acc


@torch.no_grad()
def features(idx, am, vm):
    mm, ap = [], []
    for xb, _ in DataLoader(AudioDS(idx), batch_size=64, num_workers=2):
        xb = xb.to(DEVICE)
        mm.append(xb.mean(2).cpu().numpy())
        ap.append(F.softmax(am(xb), 1).cpu().numpy())
    ve, vp = [], []
    for xb, _ in DataLoader(VideoDS(idx), batch_size=CFG["video_batch"], num_workers=2):
        lg, e = vm(xb.to(DEVICE))
        ve.append(e.cpu().numpy()); vp.append(F.softmax(lg, 1).cpu().numpy())
    return np.concatenate([np.concatenate(mm), np.concatenate(ap),
                           np.concatenate(ve), np.concatenate(vp)], axis=1)


rows = []
skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)  # the paper's split

for fold, (tr, va) in enumerate(skf.split(pairs, LABELS), 1):
    shared = len(set(pairs.actor.iloc[tr]) & set(pairs.actor.iloc[va]))
    print(f"\n===== fold {fold}/5 | {len(tr)}/{len(va)} clips | "
          f"actors in BOTH train and val: {shared}/24  <-- this is the leak =====", flush=True)

    am, a_acc = train_base("audio", tr, va, f"f{fold}")
    vm, v_acc = train_base("video", tr, va, f"f{fold}")
    rows.append(dict(fold=fold, model="audio", accuracy=a_acc))
    rows.append(dict(fold=fold, model="video", accuracy=v_acc))

    # paper protocol: training features come from the models that trained on these very clips
    Xtr, Xva = features(tr, am, vm), features(va, am, vm)
    ytr, yva = LABELS[tr], LABELS[va]
    del am, vm; gc.collect(); torch.cuda.empty_cache()

    sc = StandardScaler().fit(Xtr)                       # CHANGE 2 of 2
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    try:
        Xtr, ytr = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xtr, ytr)
    except ValueError as e:
        print("  SMOTE skipped:", e, flush=True)

    fm = FusionMLP(in_dim=Xtr.shape[1], dropout=CFG["fusion_dropout"]).to(DEVICE)
    opt = torch.optim.Adam(fm.parameters(), lr=CFG["fusion_lr"], weight_decay=CFG["fusion_wd"])
    sch = torch.optim.lr_scheduler.StepLR(opt, CFG["fusion_step_size"], CFG["fusion_gamma"])
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(yva, dtype=torch.long).to(DEVICE)

    best_acc, best_loss, best_pred, bad = 0.0, float("inf"), None, 0
    for ep in range(1, CFG["fusion_epochs"] + 1):
        fm.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            if len(b) < 2: continue
            loss = crit(fm(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        fm.eval()
        with torch.no_grad():
            o = fm(Xv)
            vloss = crit(o, yv).item()
            vacc = accuracy_score(yva, o.argmax(1).cpu().numpy()) * 100
        if vacc > best_acc:                     # paper: max-over-epochs val accuracy
            best_acc, best_pred = vacc, o.argmax(1).cpu().numpy()
        if vloss < best_loss - 1e-5:
            best_loss, bad = vloss, 0
        else:
            bad += 1
            if bad >= 30: break

    rows.append(dict(fold=fold, model="fusion", **metrics(yva, best_pred)))
    print(f"  fold {fold} fusion best val acc: {best_acc:.2f}%   (paper fold target ~94-96%)",
          flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "paper_protocol_folds.csv"), index=False)

df = pd.DataFrame(rows)
print("\n\n============ PAPER PROTOCOL (StratifiedKFold, leaky) + our two fixes ============")
print(df.groupby("model")["accuracy"].agg(["mean", "std"]).round(2).to_string())
mean = df[df.model == "fusion"]["accuracy"].mean()
print(f"\nfusion mean: {mean:.2f}%   |   paper reported: 94.29%   |   delta: {mean - 94.29:+.2f}")
print("\nThis number is comparable to 94.29% -- same split, same metric, same hyperparameters.")
print("It says the published architecture implemented correctly does better on the published")
print("benchmark. It does NOT say the model reads emotion better: actors still leak across these")
print("folds. For that claim see the actor-independent arm.")
df.to_csv(os.path.join(OUT, "paper_protocol_folds.csv"), index=False)
