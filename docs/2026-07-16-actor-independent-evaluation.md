# Actor-independent re-evaluation + binary affect head

**Date:** 2026-07-16 · **Branch:** `actor-independent-eval` · **Deadline:** 2026-07-17

Brief: improve accuracy while being careful of overfitting, and add a depressed/not-depressed
output. Both turned out to depend on a prior question — whether the reported accuracy measures
emotion at all.

## 1. The finding that reframes the task

`paper-notebook.ipynb` reports **94.29%** fusion accuracy (paper: 96.06%). That number is
inflated by **actor leakage**.

Cell 27 splits with `StratifiedKFold(n_splits=5, shuffle=True)` over `pairs_df` — a random split
across 2,452 clips from 24 actors. Every actor in a validation fold also appears in that fold's
training set. This is worse than ordinary speaker leakage: RAVDESS uses two fixed sentences and
records each (actor, emotion, intensity, statement) **twice**, so repetition 1 routinely lands in
train while repetition 2 lands in val — same face, same voice, same sentence, same emotion, same
intensity. The model can score well by recognising actors rather than reading emotion.

Corroborating signal: the **video-only** model reaches **91.24%** on fold 1 (cell 31 output).
Video-only emotion recognition at 91% is not plausible from facial expression alone. That is the
memorisation signature showing up directly.

The standard RAVDESS protocol is actor-independent (group folds by actor / leave-one-speaker-out).
Switching **will lower the number** — plausibly into the 55–75% band for fusion. That is not a
regression introduced by this work; it is the first honest measurement of the model.

## 2. Verified vs. inferred

Everything below was checked against the data or the code, not asserted from reading:

| claim | status | evidence |
|---|---|---|
| Actor leakage in the split | **verified** | cell 27, `StratifiedKFold(shuffle=True)` over clips |
| Fusion train features are self-predictions | **verified** | cell 35 line 809 passes `train_df` through `audio_models[fold-1]` |
| Metric is max-over-epochs on the scored set | **verified** | cells 832→839→840: best-val checkpoint re-scored on that same val loader |
| 4,904 videos vs 2,452 audio; key collides | **verified** | Kaggle file listing: 2,452 each of modality 01/02/03 |
| Fused features unscaled | **verified** | cell 33 concatenates raw MFCC means (≈ −300) with 0–1 softmax; `FusionMLP`'s first `BatchNorm` sits *after* the first `Linear` |
| Video norm mismatches pretrained weights | **verified as a mismatch, NOT as a defect** | cell 15 standardises per clip; `R3D_18_Weights.KINETICS400_V1` expects Kinetics stats. See §2.1 — correcting it appears to *hurt*. |
| Cell 5's "Song only" comment is wrong | **verified** | 1,440 speech + 1,012 song = 2,452; disgust/surprised (192 each) exist only in speech |
| Dataset is 25.6 GB | **verified** | `orvile/ravdess-dataset` = 25,615,024,242 bytes |

### 2.1 The Kinetics normalization change may be a regression

Paper-arm fold 1, measured 2026-07-17, against the original notebook's committed fold-1 outputs
on the same split and seed:

| model | ours | original | delta |
|---|---|---|---|
| audio | 78.21 | 80.65 | −2.4 |
| video | 85.54 | 91.24 | **−5.7** |
| fusion | **94.70** | 94.29 (5-fold mean) | **+0.41** |

The Kinetics change was introduced on the theory that feeding Kinetics-pretrained weights
per-clip-standardized inputs "fights" them. Video came out **5.7 points worse**. A plausible
mechanism running the other way: per-clip standardization removes per-clip lighting and contrast
variation, and RAVDESS records each actor under consistent studio conditions, so normalizing the
nuisance variation away may matter more here than matching Kinetics' global channel statistics.

**This is not yet attributable.** Audio dropped 2.4 points on the same fold and *audio was not
changed at all* — so roughly 2.4 points of run-to-run nondeterminism (cuDNN autotuning, dropout,
dataloader order) is in play. Video's 5.7 exceeds that but not decisively.

**Do not claim Kinetics normalization as an improvement in the paper on this evidence.** Settling
it needs one ablation: per-clip normalization, everything else held fixed. That did not fit the
one-run budget.

Worth noting regardless: fusion improved *despite* being fed worse video features, which suggests
the StandardScaler contributes more than the +0.41 headline implies — and that the two changes
should have been measured separately rather than bundled. Bundling them was a scoping decision
made against the deadline (§5); this is its cost.

### Corrections to earlier analysis

Two claims from the first pass did not survive verification, and are recorded here so they don't
get cited as findings:

- **The key collision is cosmetic, not an accuracy bug.** It is real — `key` omits `modality`, so
  each video key matches both the full-AV (01) and video-only (02) copy, and `v.iloc[0]` resolves
  it by glob order. But 01 and 02 are the *same visual recording*; whichever is chosen yields
  identical frames. It is a determinism wart. Fixed because it is free, not because it moves the
  number.
- **The `refine` head is a deviation, not a bug.** R3D-18's own pooling has already collapsed the
  volume to 1×1×1 before `Conv3d(512,256,k=3,padding=1)` runs, so those convs act on padding and
  reduce to their centre taps — wasting 26/27 of their parameters. But cell 21's own comment shows
  the author knew and chose it deliberately. It is a documented departure from the paper's Fig. 4,
  left untouched (see §5).

## 3. Decisions

**Report three rows, not two.** The task has two legitimate but different questions in it, and
collapsing them into one number is what produced the original problem.

| # | protocol | what it answers |
|---|---|---|
| 1 | paper's split, paper's code — **94.29%**, cited | the published claim |
| 2 | paper's split, our two fixes — `src/paper_protocol.py` | *did we improve the published architecture?* |
| 3 | actor-independent, our fixes — `src/train_actor_independent.py` | *does it read emotion at all?* |

Row 2 exists because "beat the paper" is a real requirement and there is an honest way to satisfy
it: **same split, same metric, same hyperparameters, better implementation.** It is directly
comparable to 94.29% and defensible as *"the published architecture, implemented correctly, on the
published benchmark."* Kinetics normalization alone should be a genuine gain — the original feeds
Kinetics-pretrained weights inputs from the wrong distribution.

Row 2 is explicitly **not** a claim that the model reads emotion better. Actors still leak across
those folds. Conflating rows 2 and 3 would reproduce exactly the error this document exists to
document.

Row 3 is the contribution: *the published 96.06% depends on speaker leakage; under
actor-independent evaluation the same architecture gets Y%.* A reproduction that finds a
methodological flaw is stronger than a tuning delta, and it survives a reviewer running the
obvious check.

Row 1 is **cited from the committed notebook outputs, not re-run** — those numbers already exist.
Row 2 supersedes the earlier plan to cite-only, because it answers a question citation cannot.

**Fusion features become out-of-fold (standard stacking).** Inner 2-fold GroupKFold over the
training actors produces clean features for every training clip; base models retrained on the full
training set produce the test features. This is what `sklearn`'s `StackingClassifier` does.

An earlier plan — averaging two half-models' outputs for the test features — was **discarded as
wrong**: the 568-dim vector contains a 512-dim R3D-18 embedding, the backbone is fully fine-tuned
(cell 21 freezes nothing), and embeddings from two independently fine-tuned networks occupy
unrelated coordinate spaces. Averaging them is arithmetic on incommensurable vectors.

The leaky arm costs nothing extra: the full-train models predicting their own training clips
reproduce the original protocol exactly, using inference alone. So the run yields an ablation
where **only the fusion features differ**, on identical held-out actors.

**Binary output is post-hoc, not a trained head.** The training run persists per-sample
out-of-fold predictions, so a mapping is a set of emotion names applied downstream. Every
candidate mapping is reportable from one run, and revisiting the choice costs no compute.

## 4. On the depressed/not-depressed label

RAVDESS contains **no depression label** — it is 24 actors portraying 8 emotions on two fixed
sentences. `sad → depressed` builds a detector for *acted sadness*. That is wrong in both
directions: acted sadness is not clinical depression, and depression frequently presents as
blunted, flat affect rather than sad affect. Depression is a clinical construct measured over
weeks (PHQ-8, BDI), not a three-second expression.

The class counts make this concrete, independent of any argument about validity:

| mapping | positive | always-predict-negative scores |
|---|---|---|
| `sad → depressed` | 15.3% | **84.7%** |
| `sad + fearful → depressed` | 30.7% | **69.3%** |
| `sad, angry, fearful, disgust → negative valence` | 53.8% | **53.8%** |

A `sad → depressed` model reporting ~90% is beating a hardcoded constant by five points. The
imbalance manufactures the number. Only the valence grouping is balanced enough for accuracy to
carry information — so the balanced grouping and the honestly-nameable grouping are the *same
grouping*, and no rigour is traded for the deadline.

**Therefore:** `negative_valence` is the headline. The `depressed_*` rows are reported alongside
it with balanced accuracy, macro-F1, AUC and their majority baseline in the same table, so the
caveat is visible in the numbers rather than buried in prose.

The two-stage plan (pretrain RAVDESS → fine-tune DAIC-WOZ on real PHQ-8 labels) is **cut**:
DAIC-WOZ requires a signed access agreement with days of lead time, which the 2026-07-17 deadline
rules out. The paper reports the transfer pathway as demonstrated with clinical validation pending
data access. If the deadline moves, file the request first.

## 5. Cut from scope

- **The `refine` head fix.** Slicing R3D-18 before its avgpool so the 3D convs see a real
  spatiotemporal map is the right architecture, but it is a change needing re-tuning, and the
  budget is one run. If it underperformed there would be no second attempt. Known-issue/future-work.
- **Full nested CV** for the fusion features (~3× cost). Inner 2-fold is the affordable
  approximation.

## 6. Protocol

```
outer GroupKFold(5) by actor          -- no actor in both train and test
  inner GroupKFold(2) by train actors -- out-of-fold features for every training clip
  full-train base models              -- test features + standalone rows + leaky-arm features
  fusion x2 (leaky | out-of-fold)     -- identical test fold; only the features differ
```

Early stopping for every model uses actors carved from the **training** side
(`GroupShuffleSplit`); the test fold is never consulted for checkpoint selection. This removes the
max-over-epochs inflation. SMOTE runs **after** the train/val split and only on the training
portion — applying it first would synthesize training points by interpolating validation points.

## 7. Files

| file | role |
|---|---|
| `src/cache_faces.py` | one-time face extraction → uint8 cache (Kaggle, CPU) |
| `src/paper_protocol.py` | row 2 — paper's protocol + the two fixes (Kaggle, GPU) |
| `src/train_actor_independent.py` | row 3 — the protocol above (Kaggle, GPU) |
| `src/analyze_binary_mapping.py` | post-hoc binary mapping over saved predictions |
| `results/` | fold results, summary, per-sample out-of-fold predictions |

Heavy compute lives in scripts, not notebooks: `.ipynb` diffs are unreadable JSON, which is why
this repo's history is a series of opaque "Add files via upload" commits against one 138 KB file.
Presentation belongs in a notebook that loads the saved CSVs — it re-runs in seconds because the
4-hour compute already happened.

The face cache stores **uint8 raw crops**, not the original's float32-with-normalization-baked-in.
It is 4× smaller (~1.5 GB vs ~5.9 GB) and keeps normalization a free parameter, so changing it
does not mean re-extracting 2,452 clips.
