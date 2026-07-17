"""
Collapse the 8-class fusion predictions to a binary output, post hoc.

No retraining and no second model: this reads the per-sample out-of-fold predictions the
training run already wrote, so a mapping is just a set of emotion names. That means every
candidate mapping can be reported from one run, and changing your mind costs nothing.

Read the baseline column before the accuracy column. RAVDESS class counts are lopsided, so a
mapping's accuracy is only meaningful relative to what a constant classifier scores on it:

    sad -> depressed        376/2452 = 15.3% positive  -> always-negative scores 84.7%
    sad+fearful             752/2452 = 30.7% positive  -> always-negative scores 69.3%
    negative valence       1320/2452 = 53.8% positive  -> always-negative scores 53.8%

Only the valence grouping is balanced enough for raw accuracy to carry information. The other
two are reported with balanced accuracy, macro-F1 and AUC precisely because their headline
accuracy is mostly the class prior talking.

Naming: RAVDESS contains no depression label. It is 24 actors portraying emotions on two fixed
sentences. The `depressed_*` rows below detect *acted sadness*, which is not clinical depression
in either direction -- acted sadness is not depression, and depression frequently presents as
blunted affect rather than sad affect. They are included because they were asked for, and are
named `depressed_*` only to match that request; the defensible headline is `negative_valence`.
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
IDX = {e: i for i, e in enumerate(EMOTIONS)}

MAPPINGS = {
    # name -> set of emotions treated as the positive class
    "negative_valence": {"sad", "angry", "fearful", "disgust"},
    "depressed_sad_only": {"sad"},
    "depressed_sad_fearful": {"sad", "fearful"},
}

SRC = next(iter(glob.glob("**/oof_predictions.csv", recursive=True)
                + glob.glob("/kaggle/input/**/oof_predictions.csv", recursive=True)), None)
assert SRC, "oof_predictions.csv not found -- run the training kernel first"
df = pd.read_csv(SRC)
print(f"{len(df)} predictions from {df.fold.nunique()} folds, {df.actor.nunique()} actors\n")

for source in ["fusion", "fusion_paper", "audio", "video"]:
    cols = [f"{source}_p{i}" for i in range(8)]
    if not all(c in df.columns for c in cols):
        continue
    P = df[cols].values                      # (N, 8) softmax
    y8 = df["true"].values

    print(f"================ {source} ================")
    rows = []
    for name, pos in MAPPINGS.items():
        pos_idx = [IDX[e] for e in pos]
        y = np.isin(y8, pos_idx).astype(int)

        # Probability-sum uses the whole distribution; argmax-then-map throws away everything
        # except the winner. Reported side by side because the gap is worth seeing.
        score = P[:, pos_idx].sum(1)
        pred_prob = (score >= 0.5).astype(int)
        pred_argmax = np.isin(P.argmax(1), pos_idx).astype(int)

        base = max(y.mean(), 1 - y.mean()) * 100   # always-predict-majority
        rows.append(dict(
            mapping=name,
            positive_pct=round(y.mean() * 100, 1),
            majority_baseline=round(base, 2),
            acc_probsum=round(accuracy_score(y, pred_prob) * 100, 2),
            acc_argmax=round(accuracy_score(y, pred_argmax) * 100, 2),
            balanced_acc=round(balanced_accuracy_score(y, pred_prob) * 100, 2),
            f1=round(f1_score(y, pred_prob, zero_division=0) * 100, 2),
            precision=round(precision_score(y, pred_prob, zero_division=0) * 100, 2),
            recall=round(recall_score(y, pred_prob, zero_division=0) * 100, 2),
            auc=round(roc_auc_score(y, score) * 100, 2),
            over_baseline=round(accuracy_score(y, pred_prob) * 100 - base, 2),
        ))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    for r in rows:
        if r["over_baseline"] <= 0:
            print(f"\n  !! {r['mapping']}: {r['acc_probsum']}% accuracy is at or BELOW the "
                  f"{r['majority_baseline']}% majority baseline -- this mapping learns nothing.")
    print()

# Confusion matrix for the headline mapping only.
cols = [f"fusion_p{i}" for i in range(8)]
if all(c in df.columns for c in cols):
    pos_idx = [IDX[e] for e in MAPPINGS["negative_valence"]]
    y = np.isin(df["true"].values, pos_idx).astype(int)
    p = (df[cols].values[:, pos_idx].sum(1) >= 0.5).astype(int)
    cm = confusion_matrix(y, p)
    print("negative_valence confusion (fusion, out-of-fold):")
    print(pd.DataFrame(cm, index=["true_nonneg", "true_negative"],
                       columns=["pred_nonneg", "pred_negative"]).to_string())

    # Per-emotion recall shows which emotions the binary head actually keys on -- if
    # "depressed" fires mostly on angry, the label is doing no work.
    print("\nper-emotion positive rate under negative_valence (fusion):")
    sub = pd.DataFrame({"emotion": [EMOTIONS[i] for i in df["true"]], "pred_pos": p})
    print(sub.groupby("emotion")["pred_pos"].agg(["mean", "count"]).round(3).to_string())

os.makedirs("results", exist_ok=True)
print("\nnote: mappings are declarative -- edit MAPPINGS and re-run; no retraining involved.")
