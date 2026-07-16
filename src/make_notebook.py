"""Generate results.ipynb. Kept as a generator so the cell sources stay reviewable in git
rather than living as escaped strings inside notebook JSON."""
import json, sys

MD, CODE = "markdown", "code"

CELLS = [
(MD, r"""# Actor-independent re-evaluation — results

Loads the CSVs written by `src/paper_protocol.py` and `src/train_actor_independent.py`.
No training happens here: the ~4 hours of compute already ran on Kaggle, so this notebook
re-executes in seconds.

**Three rows, answering two different questions.** Collapsing them into one number is what
produced the problem this work documents.

| # | protocol | question |
|---|---|---|
| 1 | paper's split, paper's code (cited) | the published claim |
| 2 | paper's split, our two fixes | *did we improve the published architecture?* |
| 3 | actor-independent, our fixes | *does it read emotion at all?* |
"""),

(CODE, r"""import glob, os
import numpy as np, pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

def find(name):
    hits = glob.glob(f"results/**/{name}", recursive=True) + glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

PAPER_REPORTED = 94.29   # from the committed notebook outputs; not re-run

paper = pd.read_csv(find("paper_protocol_folds.csv")) if find("paper_protocol_folds.csv") else None
folds = pd.read_csv(find("fold_results.csv")) if find("fold_results.csv") else None
preds = pd.read_csv(find("oof_predictions.csv")) if find("oof_predictions.csv") else None

for n, d in [("paper_protocol_folds", paper), ("fold_results", folds), ("oof_predictions", preds)]:
    print(f"{n:22} {'MISSING' if d is None else str(d.shape)}")"""),

(MD, r"""## The headline

Row 2 is the "beat the paper" number: **same split, same metric, same hyperparameters** — only
the two implementation fixes differ (Kinetics normalization matching the pretrained weights, and
a StandardScaler on the fused vector). It is directly comparable to 94.29%.

Row 3 is the contribution. It is **not** comparable to rows 1–2 and is not supposed to be: it
answers a harder question."""),

(CODE, r"""rows = [dict(protocol="1. paper split, paper code (cited)", fusion=PAPER_REPORTED, comparable_to_paper="yes")]

if paper is not None:
    m = paper[paper.model == "fusion"]["accuracy"]
    rows.append(dict(protocol="2. paper split, our fixes", fusion=round(m.mean(), 2),
                     std=round(m.std(), 2), comparable_to_paper="yes",
                     delta_vs_paper=round(m.mean() - PAPER_REPORTED, 2)))

if folds is not None:
    for name, label in [("fusion-paper", "3a. actor-independent, paper's fusion features"),
                        ("fusion-oof", "3b. actor-independent, out-of-fold fusion features")]:
        m = folds[folds.model == name]["accuracy"]
        if len(m):
            rows.append(dict(protocol=label, fusion=round(m.mean(), 2), std=round(m.std(), 2),
                             comparable_to_paper="NO -- different task"))

display(pd.DataFrame(rows).fillna(""))"""),

(MD, r"""### Reading this table

- **Row 2 vs row 1** — whether the published architecture, implemented correctly, does better on
  the published benchmark. A legitimate improvement claim.
- **Row 2 vs row 3** — the size of the actor-leakage effect. Actors appear in both train and
  validation in rows 1–2; in row 3 they never do.
- **Row 3a vs row 3b** — the fusion-feature leak in isolation, on identical held-out actors. 3a
  trains the fusion MLP on features from models that memorized those clips (the paper's protocol);
  3b uses out-of-fold features.

Row 2 does **not** say the model reads emotion better. Actors still leak across those folds.
Only row 3 speaks to generalization."""),

(CODE, r"""if folds is not None:
    piv = folds.pivot_table(index="model", values=["accuracy", "balanced_accuracy", "f1"],
                            aggfunc=["mean", "std"]).round(2)
    print("actor-independent, per-model:")
    display(piv)
    print("\nper-fold (each fold = a disjoint set of held-out actors):")
    display(folds.pivot_table(index="fold", columns="model", values="accuracy").round(2))"""),

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

- **Row 1 was not re-run.** 94.29% is cited from the committed notebook outputs.
- **`refine` head left untouched.** R3D-18's pooling already collapses the volume to 1×1×1 before
  those 3×3×3 convs run, so they act on padding and reduce to their centre taps — wasting 26/27 of
  their parameters. Cell 21's own comment shows this was deliberate, so it is a documented
  deviation from the paper's Fig. 4, not a bug. Fixing it is an architecture change needing
  re-tuning, and the budget was one run.
- **Out-of-fold features use inner 2-fold**, not full nested CV — the affordable approximation.
  Train features come from models trained on half the training actors; test features from models
  retrained on all of them (standard stacking, as `sklearn`'s `StackingClassifier` does).
- **The key-collision fix is cosmetic.** Every video key matched 2 files (2452/2452 confirmed), but
  the 01 and 02 copies are the same footage, so it buys determinism, not accuracy.
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
