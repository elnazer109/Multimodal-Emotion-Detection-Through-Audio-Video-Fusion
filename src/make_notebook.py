"""Generate results.ipynb. Kept as a generator so the cell sources stay reviewable in git
rather than living as escaped strings inside notebook JSON."""
import json, sys

MD, CODE = "markdown", "code"

CELLS = [
(MD, r"""# Emotion Unlocked — reproduction, correction, and an honest re-evaluation

Loads the CSVs the Kaggle runs wrote. No training happens here — the compute already ran, so this
re-executes in seconds.

**Two questions, deliberately kept apart.** Collapsing them into one number is what produced the
problem this work documents.

| question | protocol | answered by |
|---|---|---|
| *Did we improve the published architecture?* | the paper's own split | the arms below, vs **96.06%** |
| *Does it read emotion from an unseen person?* | actor-independent GroupKFold | the honest section |

Two things the paper's own text establishes, both load-bearing:

1. **§F Cross-Validation Strategy** — *"the database is split into five subsets… four for
   training, one for validation."* Split into subsets **of samples**, with no mention of actors or
   speakers. RAVDESS has only 24 actors delivering two fixed sentences, each recorded twice — so
   repetition 1 trains while repetition 2 validates: same face, same voice, same sentence. **The
   leak is in the publication, not just in the notebook.**
2. **§VI.6** — *"The dataset has 1,360 distinct samples."* RAVDESS pairs to 2,452 (1,440 speech +
   1,012 song), or 1,440 speech-only. **Neither is 1,360**, and the notebook trains on 2,452 —
   including the song subset, which contains no *disgust* or *surprised* at all.
"""),

(CODE, r"""import glob, os
import numpy as np, pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

def find(name):
    hits = glob.glob(f"results/**/{name}", recursive=True) + glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

# The PAPER's Table VII -- read from video paper.pdf, not from the notebook's outputs.
# The notebook scored 94.29 and never reproduced its own paper, falling 1.77 short.
PAPER = {"audio": 79.24, "video": 92.72, "fusion": 96.06}
NOTEBOOK_FUSION = 94.29
PAPER_FOLDS = pd.DataFrame({
    "fold":   [1, 2, 3, 4, 5],
    "audio":  [78.8, 80.6, 77.4, 79.8, 79.6],
    "video":  [93.0, 91.9, 93.7, 96.6, 88.2],
    "fusion": [95.6, 97.6, 97.2, 95.3, 94.6],
})

RUNS = {  # arm -> its fold-results csv
    "kinetics (regression)": "paper_protocol_folds.csv",
    "per-clip + scaler":     "paper_perclip_folds.csv",
    "+ refine head":         "paper_refine_folds.csv",
}
arms = {k: pd.read_csv(find(v)) for k, v in RUNS.items() if find(v)}
folds = pd.read_csv(find("fold_results.csv")) if find("fold_results.csv") else None
preds = pd.read_csv(find("oof_predictions.csv")) if find("oof_predictions.csv") else None

print("paper Table VII:", PAPER)
for k in RUNS:
    print(f"{k:24} {'MISSING' if k not in arms else str(arms[k].shape)}")
print(f"{'fold_results (honest)':24} {'MISSING' if folds is None else str(folds.shape)}")
print(f"{'oof_predictions':24} {'MISSING' if preds is None else str(preds.shape)}")"""),

(MD, r"""## The headline

All arms below use the **paper's own protocol**: `StratifiedKFold(shuffle=True)`, its
max-over-epochs metric, its hyperparameters. So they are directly comparable to Table VII's
**96.06%**.

Note the target. `94.29%` is what the *notebook* scored — it never reproduced its own paper,
falling 1.77 short. The bar is 96.06."""),

(CODE, r"""rows = [
    dict(arm="paper, Table VII (published)", **PAPER, note="the bar"),
    dict(arm="notebook, as committed", fusion=NOTEBOOK_FUSION,
         note="never reproduced its own paper: -1.77"),
]
for name, d in arms.items():
    piv = d.pivot_table(index="fold", columns="model", values="accuracy")
    r = dict(arm=name, note="")
    for m in ["audio", "video", "fusion"]:
        if m in piv:
            r[m] = round(piv[m].mean(), 2)
    if "fusion" in r:
        r["note"] = f"vs paper {r['fusion'] - PAPER['fusion']:+.2f}"
    rows.append(r)

t = pd.DataFrame(rows)[["arm", "audio", "video", "fusion", "note"]]
display(t.fillna("").style.hide(axis="index")
         .set_caption("all rows on the paper's protocol -> comparable to 96.06"))

for name, d in arms.items():
    f = d[d.model == "fusion"]["accuracy"]
    if len(f) == 5:
        verdict = "BEATS THE PAPER" if f.mean() > PAPER["fusion"] else "short of the paper"
        print(f"{name:24} fusion {f.mean():6.2f} +-{f.std():.2f}   -> {verdict}")"""),

(MD, r"""### What the arms mean

- **kinetics** — replaced the paper's per-clip video standardization with Kinetics channel stats,
  on the theory that the original "fights" the pretrained weights. **A regression**: video fell to
  ~82 (std 8.11, folds ranging 71→89) and fusion landed *below* even the notebook. The theory was
  backwards — per-clip standardization removes lighting/contrast variation, and RAVDESS records
  each actor under consistent studio conditions.
- **per-clip + scaler** — the paper's normalization restored, keeping only the StandardScaler on
  the 568-dim fused vector (raw MFCC means sit near −300 beside 0–1 softmax probabilities, and
  `FusionMLP`'s first BatchNorm is *after* the first Linear, so nothing normalized the input).
- **+ refine head** — additionally implements the paper's Fig. 4. Sec. VII.2 specifies
  r3d_18 → stacked 3D convs → global average pooling. The notebook slices `children()[:-1]`,
  keeping r3d_18's own avgpool, so the volume is already 1×1×1 and each 3×3×3 conv degenerates to
  its centre tap. Slicing `[:-2]` emits (B,512,2,7,7) and the convs do real spatiotemporal work.

Every arm still has **actors leaking across folds** — that's the paper's protocol, kept
deliberately so the comparison is like-for-like. None of these numbers says the model reads
emotion from an unseen person. That question is below."""),

(CODE, r"""# per-fold against the paper's own Table VII
for name, d in arms.items():
    piv = d.pivot_table(index="fold", columns="model", values="accuracy").round(2)
    cmp = piv.join(PAPER_FOLDS.set_index("fold"), rsuffix="_paper")
    for m in ["audio", "video", "fusion"]:
        if m in piv:
            cmp[f"{m}_delta"] = (piv[m] - PAPER_FOLDS.set_index("fold")[m]).round(2)
    print(f"--- {name} vs paper Table VII, per fold ---")
    display(cmp[[c for c in cmp.columns if "delta" in c or c in ("audio","video","fusion")]])"""),

(MD, """## The honest number

Everything above lets actors leak. This is what the same architecture scores when the people in
the test fold were never seen in training — the question the paper never asks."""),

(CODE, r"""if folds is not None:
    piv = folds.pivot_table(index="model", values=["accuracy", "balanced_accuracy", "f1"],
                            aggfunc=["mean", "std"]).round(2)
    print("actor-independent, per-model:")
    display(piv)
    print("\nper-fold (each fold = a disjoint set of held-out actors):")
    display(folds.pivot_table(index="fold", columns="model", values="accuracy").round(2))
    fo = folds[folds.model == "fusion-oof"]["accuracy"]
    fp = folds[folds.model == "fusion-paper"]["accuracy"]
    if len(fo):
        print(f"\nactor-independent fusion (out-of-fold features): {fo.mean():.2f}")
        print(f"paper's headline, actors leaking:                 {PAPER['fusion']:.2f}")
        print(f"cost of removing the leak:                        {fo.mean() - PAPER['fusion']:+.2f}")
    if len(fo) and len(fp):
        print(f"\nfusion-feature leak alone (same held-out actors):  {fp.mean() - fo.mean():+.2f}")"""),

(MD, """## Where the errors go

If the model were reading emotion, confusions should cluster by affective similarity
(sad/calm, angry/fearful). Confusions that instead track actor or vocal channel would say
something else is driving it."""),

(CODE, r"""EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

if preds is not None:
    from sklearn.metrics import confusion_matrix
    P = preds[[f"fusion_p{i}" for i in range(8)]].values
    cm = confusion_matrix(preds["true"], P.argmax(1))
    cmn = (cm / cm.sum(1, keepdims=True) * 100).round(1)
    display(pd.DataFrame(cmn, index=EMOTIONS, columns=EMOTIONS)
              .style.background_gradient(cmap="Blues", axis=None)
              .format("{:.1f}").set_caption("row-normalised % (rows = true label)"))

    print("\naccuracy by vocal channel (1=speech, 2=song):")
    d = preds.assign(correct=(P.argmax(1) == preds["true"]))
    display(d.groupby("vocal_channel")["correct"].agg(["mean", "count"]).round(3))"""),

(MD, r"""## Binary output: depressed / not-depressed

Post-hoc, from the saved 8-class predictions. **No retraining** — a mapping is just a set of
emotion names, so every candidate is reportable from the same run.

**Read the baseline column before the accuracy column.** RAVDESS is lopsided, so a mapping's
accuracy only means something relative to a constant classifier:

| mapping | positive | always-predict-negative scores |
|---|---|---|
| `sad → depressed` | 15.3% | **84.7%** |
| `sad + fearful` | 30.7% | **69.3%** |
| negative valence | 53.8% | **53.8%** |

Only the valence grouping is balanced enough for raw accuracy to carry information — which is why
the balanced grouping and the honestly-nameable grouping turn out to be the same grouping.

RAVDESS contains **no depression label**. It is 24 actors portraying emotions on two fixed
sentences. The `depressed_*` rows detect *acted sadness*, which is not clinical depression in
either direction — acted sadness is not depression, and depression frequently presents as blunted
affect rather than sad affect. They are shown because they were asked for; the defensible headline
is `negative_valence`."""),

(CODE, r"""from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
IDX = {e: i for i, e in enumerate(EMOTIONS)}
MAPPINGS = {
    "negative_valence":      {"sad", "angry", "fearful", "disgust"},
    "depressed_sad_only":    {"sad"},
    "depressed_sad_fearful": {"sad", "fearful"},
}

if preds is not None:
    P = preds[[f"fusion_p{i}" for i in range(8)]].values
    y8 = preds["true"].values
    out = []
    for name, pos in MAPPINGS.items():
        pi = [IDX[e] for e in pos]
        y = np.isin(y8, pi).astype(int)
        score = P[:, pi].sum(1)          # whole distribution, not just the argmax
        pred = (score >= 0.5).astype(int)
        base = max(y.mean(), 1 - y.mean()) * 100
        acc = accuracy_score(y, pred) * 100
        out.append(dict(mapping=name, positive_pct=round(y.mean()*100, 1),
                        majority_baseline=round(base, 2), accuracy=round(acc, 2),
                        over_baseline=round(acc - base, 2),
                        balanced_acc=round(balanced_accuracy_score(y, pred)*100, 2),
                        f1=round(f1_score(y, pred, zero_division=0)*100, 2),
                        auc=round(roc_auc_score(y, score)*100, 2)))
    t = pd.DataFrame(out)
    display(t.style.bar(subset=["over_baseline"], align="zero", color=["#d65f5f", "#5fba7d"])
             .format(precision=2))
    for r in out:
        if r["over_baseline"] <= 0:
            print(f"!! {r['mapping']}: {r['accuracy']}% is at or BELOW its "
                  f"{r['majority_baseline']}% baseline -- this mapping learns nothing.")"""),

(MD, """### Which emotions does the binary head actually fire on?

If a "depressed" head fires mostly on *angry*, the label is doing no work regardless of its
accuracy."""),

(CODE, r"""if preds is not None:
    pi = [IDX[e] for e in MAPPINGS["negative_valence"]]
    p = (P[:, pi].sum(1) >= 0.5).astype(int)
    d = pd.DataFrame({"true_emotion": [EMOTIONS[i] for i in y8], "predicted_positive": p})
    display(d.groupby("true_emotion")["predicted_positive"]
             .agg(["mean", "count"]).round(3)
             .rename(columns={"mean": "fires_positive_rate"})
             .sort_values("fires_positive_rate", ascending=False))"""),

(MD, r"""## Caveats

- **The bar is the PAPER's 96.06%, not the notebook's 94.29%.** Table VII of `video paper.pdf`
  reports audio 79.24 / video 92.72 / fusion 96.06. The committed notebook scores 94.29, so it
  never reproduced its own paper. Earlier drafts of this work benchmarked against 94.29 by mistake.
- **The paper's own sample count does not reconcile.** Sec. VI.6 states "1,360 distinct samples,
  each with both an audio and video recording". RAVDESS pairs to 2,452 (1,440 speech + 1,012 song),
  or 1,440 speech-only. Neither is 1,360. The notebook trains on 2,452 — including the song subset,
  which contains no disgust or surprised at all — so it is not training on the paper's data.
- **The paper's Fig. 4 is not implemented by the notebook.** Sec. VII.2 specifies stacked 3D convs
  *then* global average pooling; the notebook pools first, so the convs act on a 1×1×1 volume and
  degenerate to their centre taps. Cell 21's comment shows the author knew and chose it anyway.
  The `+ refine head` arm implements the paper as written for the first time.
- **The paper does not state its epoch counts.** The notebook uses 15 for video, and its fold-1 log
  shows validation accuracy still rising at epoch 15 (90.02 → 91.24), so video is plausibly
  under-trained. Untested — the paper's max-over-epochs metric rewards longer training, so a fair
  comparison would need the epoch count the paper never published.
- **Out-of-fold features use inner 2-fold**, not full nested CV — the affordable approximation.
  Train features come from models trained on half the training actors; test features from models
  retrained on all of them (standard stacking, as `sklearn`'s `StackingClassifier` does).
- **The key-collision fix is cosmetic.** Every video key matched 2 files (2452/2452 confirmed), but
  the 01 and 02 copies are the same footage, so it buys determinism, not accuracy.
- **The metric is the paper's**: max-over-epochs accuracy on the very fold being reported. It is
  optimistic by construction. Kept anyway, because changing it would make the comparison to Table
  VII meaningless — you cannot claim to beat a number you measured differently.
- **No depression label exists in RAVDESS.** See above. The DAIC-WOZ fine-tune (real PHQ-8 labels)
  was cut for deadline; it needs a signed access agreement with days of lead time."""),
]


def cell(kind, src):
    lines = src.strip("\n").split("\n")
    src_lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == MD:
        return {"cell_type": "markdown", "metadata": {}, "source": src_lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src_lines}


nb = {
    "cells": [cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = sys.argv[1] if len(sys.argv) > 1 else "results.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"wrote {out}: {len(nb['cells'])} cells")
