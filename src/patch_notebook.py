"""Produce the corrected notebook from the original, surgically.

The team needs a notebook and there is no time to retrain. There is also no need: the runs are
finished and their per-fold results are committed as CSVs, so the notebook can carry the corrected
code AND display real measured results without executing a single epoch.

What this does:
  * patches the VideoModel cell so the paper's Fig. 4 is actually built ([:-1] -> [:-2])
  * offers GroupKFold-by-actor beside the paper's random split, as a one-line switch
  * scales the fused feature vector before the fusion MLP
  * CLEARS every stale output, because the committed outputs were produced by the OLD code and
    leaving them next to changed cells would be presenting one model's numbers as another's
  * appends a results section that loads results/*.csv -- real numbers, no training

Run:  python src/patch_notebook.py <in.ipynb> <out.ipynb>
"""
import json, sys, copy

SRC = sys.argv[1] if len(sys.argv) > 1 else "Paper_worK/paper-notebook.ipynb"
DST = sys.argv[2] if len(sys.argv) > 2 else "Paper_worK/paper-notebook-corrected.ipynb"

nb = json.load(open(SRC, encoding="utf-8"))
cells = nb["cells"]
print(f"loaded {SRC}: {len(cells)} cells")


def src_of(c):
    return "".join(c["source"])


def patch(pred, old, new, label):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = src_of(c)
        if pred(s) and old in s:
            c["source"] = (s.replace(old, new)).splitlines(keepends=True)
            print(f"  [{i:2}] patched: {label}")
            return True
    print(f"  !! NOT FOUND: {label}")
    return False


print("\npatching:")

# ---- 1. the one that matters: build the paper's Fig. 4
patch(lambda s: "class VideoModel" in s,
      "self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # ends in AdaptiveAvgPool3d -> (B,512,1,1,1)",
      "self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # stem..layer4 -> (B,512,2,7,7)",
      "VideoModel backbone slice [:-1] -> [:-2]  (build Fig. 4)")

patch(lambda s: "class VideoModel" in s,
      """        emb = self.backbone(x)                # (B, 512, 1, 1, 1)
        emb512 = emb.flatten(1)                # (B, 512)  <-- used directly in fusion

        # NOTE: the refine head expects spatial dims; R3D-18's own GAP has already
        # collapsed them to 1x1x1, so we treat emb as a (B,512,1,1,1) volume and let
        # the 3x3x3 convs act as further 1x1x1-equivalent mixing (matches Fig.4's
        # "512ch,2f,7x7 -> 256ch,2f,7x7 -> ... -> 64 features" shape-reduction spirit
        # while keeping the module chainable on the pooled embedding).
        h = self.refine(emb)""",
      """        feat = self.backbone(x)                                # (B, 512, 2, 7, 7)
        emb512 = F.adaptive_avg_pool3d(feat, 1).flatten(1)      # (B, 512)  <-- used in fusion

        # FIXED. Previously the backbone was sliced [:-1], which KEEPS R3D-18's own
        # AdaptiveAvgPool3d -- so the volume was already 1x1x1 here and every 3x3x3 conv
        # below saw 26/27 zero-padding, collapsing to its centre tap. The refine head was
        # three Linear layers wearing Conv3d costumes, and the paper's Fig.4 was never built.
        # Slicing [:-2] keeps stem..layer4, so these convs do the spatiotemporal work the
        # paper describes (Sec. VII.2: r3d_18 -> stacked 3D convs -> global average pooling).
        #   measured: video 83.24 -> 92.62  (+14.37 paired, p=0.039); paper reports 92.72
        h = self.refine(feat)""",
      "VideoModel.forward  (convs on a real 2x7x7 volume)")

# ---- 2. actor-independent split, available as a switch
patch(lambda s: "StratifiedKFold(n_splits=CFG" in s,
      """skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
fold_indices = list(skf.split(pairs_df, pairs_df["emotion_idx"]))""",
      '''# ---------------------------------------------------------------------------
# ACTOR_INDEPENDENT = False reproduces the paper's protocol (Sec. F): the dataset is
# split into five subsets of SAMPLES. Actors are never mentioned, so every actor appears
# in both training and validation. RAVDESS is 24 actors reading two fixed sentences twice
# each, so repetition 1 trains while repetition 2 validates -- same face, same voice, same
# sentence. The model can score well by recognising the actor.
#
# ACTOR_INDEPENDENT = True holds actors out entirely. The honest question.
#
#   measured, fusion:   paper protocol 96.06   ->   actor-independent 62.98
#
# Left False so the numbers stay comparable to the paper's Table VII.
# ---------------------------------------------------------------------------
ACTOR_INDEPENDENT = False

if ACTOR_INDEPENDENT:
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=CFG["n_folds"])
    fold_indices = list(gkf.split(pairs_df, groups=pairs_df["actor"]))
    for tr, va in fold_indices:
        assert not (set(pairs_df.actor.iloc[tr]) & set(pairs_df.actor.iloc[va])), "actor leaked"
else:
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(pairs_df, pairs_df["emotion_idx"]))

_shared = [len(set(pairs_df.actor.iloc[tr]) & set(pairs_df.actor.iloc[va]))
           for tr, va in fold_indices]
print(f"actors appearing in BOTH train and val, per fold: {_shared}  "
      f"({'LEAK' if max(_shared) else 'clean'})")''',
      "cell 27: GroupKFold-by-actor switch + leak counter")

# ---- 3. scale the fused vector
patch(lambda s: "SMOTE(random_state=SEED, k_neighbors=3)" in s,
      """    # SMOTE on the tabular fused features to rebalance minority emotion classes
    try:""",
      """    # Scale before SMOTE and before the MLP: the fused vector concatenates raw MFCC means
    # (~ -300) with 0..1 softmax probabilities, and FusionMLP's first BatchNorm sits AFTER
    # the first Linear, so nothing normalised the input. (Measured effect: -0.04, i.e. none.
    # Kept because it is correct, not because it helps.)
    from sklearn.preprocessing import StandardScaler
    _sc = StandardScaler().fit(Xtr)
    Xtr, Xva = _sc.transform(Xtr), _sc.transform(Xva)

    # SMOTE on the tabular fused features to rebalance minority emotion classes
    try:""",
      "cell 35: StandardScaler on the 568-d fused vector")

# ---- 4. clear stale outputs
cleared = 0
for c in cells:
    if c["cell_type"] == "code":
        if c.get("outputs"):
            cleared += 1
        c["outputs"] = []
        c["execution_count"] = None
print(f"\ncleared outputs from {cleared} cells "
      "(they were produced by the OLD code -- keeping them beside changed cells would "
      "present one model's numbers as another's)")

# ---- 5. results section, from the committed CSVs
HEAD = """# Corrected pipeline + measured results

This is `paper-notebook.ipynb` with the reproduction fixes applied. **Outputs are cleared on
purpose** — they were produced by the previous code, and showing them next to changed cells would
attribute one model's numbers to another.

**You do not need to run this to see the results.** The experiments are finished; their per-fold
accuracies are committed under `results/`, and the section at the bottom loads and displays them.

## What changed, and what it was worth

| change | effect |
|---|---|
| **`children()[:-1]` → `[:-2]`** — build the paper's Fig. 4 instead of pooling before the 3D convs | **video 83.24 → 92.62** (+14.37 paired, p=0.039) |
| `StandardScaler` on the 568-d fused vector | −0.04 — none, kept because it is correct |
| `ACTOR_INDEPENDENT` switch (cell 27) | reveals fusion 96.06 → **62.98** when actors are held out |

Full write-up: [`docs/FINDINGS.md`](../docs/FINDINGS.md) · [`docs/FINDINGS.pdf`](../docs/FINDINGS.pdf)
"""

RESULTS_MD = """---

# Measured results

Loaded from `results/` — produced by the runs in `src/`, not by executing this notebook.
"""

RESULTS_CODE = '''import glob, os
import pandas as pd, numpy as np

def _find(pat):
    for base in (".", "..", "../.."):
        hits = sorted(glob.glob(os.path.join(base, "results", "**", pat), recursive=True))
        if hits:
            return hits
    return []

PAPER = {"audio": 79.24, "video": 92.72, "fusion": 96.06}   # the paper's Table VII
ARMS = [("kinetics-arm", "Kinetics normalisation"),
        ("perclip-arm", "per-clip normalisation"),
        ("refine-arm", "+ paper's Fig. 4  <-- the fix"),
        ("epochs-arm", "+ 35 video epochs")]

rows = [dict(configuration="Paper, Table VII (published)", **PAPER, note="the bar")]
rows.append(dict(configuration="This notebook, before the fix", audio=80.67, video=83.24,
                 fusion=94.29, note="never reproduced its own paper"))
for sub, label in ARMS:
    f = _find(f"{sub}/*folds.csv")
    if not f:
        continue
    d = pd.read_csv(f[0]).pivot_table(index="fold", columns="model", values="accuracy")
    r = dict(configuration=label)
    for m in ("audio", "video", "fusion"):
        if m in d:
            r[m] = round(d[m].mean(), 2)
    r["note"] = f"vs paper {r.get('fusion', np.nan) - PAPER['fusion']:+.2f}"
    rows.append(r)

print("All rows use the PAPER's protocol -> directly comparable to Table VII\\n")
display(pd.DataFrame(rows)[["configuration", "audio", "video", "fusion", "note"]].fillna(""))

# The honest question the paper never asks.
h = _find("actor-independent*/fold_results.csv")
if h:
    d = pd.read_csv(h[0])
    g = d.groupby("model")["accuracy"].agg(["mean", "std"]).round(2)
    print("\\nActor-independent (5 actors held out per fold) -- the same architecture:")
    display(g)
    fp = g.loc["fusion-paper", "mean"] if "fusion-paper" in g.index else np.nan
    print(f"\\n  paper's headline, actors leaking : {PAPER['fusion']:.2f}")
    print(f"  actors held out                  : {fp:.2f}   ({fp - PAPER['fusion']:+.2f})")
    print("\\n  The published benchmark substantially measures actor identity.")
'''

cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": HEAD.splitlines(keepends=True)})
cells.append({"cell_type": "markdown", "metadata": {},
              "source": RESULTS_MD.splitlines(keepends=True)})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": RESULTS_CODE.splitlines(keepends=True)})

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nwrote {DST}: {len(cells)} cells")
