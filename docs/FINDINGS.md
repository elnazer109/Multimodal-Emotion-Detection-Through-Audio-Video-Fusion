# Findings — reproduction and re-evaluation of *Emotion Unlocked*

**Paper:** Vasuki, Padhma Priya M, Pooja V — *Emotion Unlocked: Multimodal Emotion Detection
Through Audio-Video Fusion*, SSN College of Engineering (preprint, SSRN 5274911).
**Branch:** `actor-independent-eval` · **Date:** 2026-07-17

Everything below is measured, not asserted. Every claim names the cell, section, or run it comes
from. Where a claim did not survive checking, it is marked as withdrawn rather than deleted.

---

## 1. The published benchmark measures actor identity

The paper reports **96.06%** fusion accuracy (Table VII). Under actor-independent evaluation the
same architecture scores **62.98%**.

| protocol | fusion | vs published |
|---|---|---|
| paper, Table VII (actors leak across folds) | **96.06** | — |
| actor-independent, paper's fusion protocol | **62.98** ± 7.96 | **−33.08** |
| actor-independent, out-of-fold fusion features | **50.25** ± 9.10 | **−45.81** |

**This is a flaw in the publication, not in anyone's implementation.** The paper's §F says:

> "five-fold cross-validation is adopted. The database is split into five subsets wherein: Four
> subsets are used for training. One subset is used for validation."

Subsets **of samples**. Actors and speakers are never mentioned anywhere in the paper. RAVDESS is
24 actors delivering **two fixed sentences**, each recorded **twice** — so repetition 1 lands in
training while repetition 2 lands in validation: same face, same voice, same sentence, same
emotion, same intensity. A model can score well by recognising *who is speaking*.

Corroborating signal, before any of our changes: the **video-only** model scores 91–93% under the
paper's protocol. Video-only emotion recognition at 92.72% from 3-second clips is not plausible
from facial expression; it is the memorisation signature showing directly.

**Two leaks, and they are additive.** The 33-point gap is actor leakage. A further **12.73**
points comes from a second, independent leak: the paper builds the fusion MLP's *training*
features by running the training clips through the models that trained on them (notebook cell 35),
so the 8-dim softmax inputs are near-perfect one-hots in training and merely ~80% right at
inference. The MLP learns to trust a signal that degrades the moment it is deployed. Measured on
*identical* held-out actors, so it is isolated from the actor leak.

**The model does read something.** Errors cluster affectively — sad↔fearful (87), angry↔disgust
(82), happy↔calm (79) — not randomly. The architecture works. It is just far weaker than
advertised.

---

## 2. The notebook never reproduced the paper

| | audio | video | fusion |
|---|---|---|---|
| paper, Table VII | 79.24 | 92.72 | **96.06** |
| `Paper_worK/paper-notebook.ipynb`, as committed | ~80.7 | 91.2 | **94.29** |

It lands **1.77 short**, and three deviations explain why it is not implementing the paper:

**a) It trains on the wrong data.** Paper §VI.6: *"The dataset has 1,360 distinct samples, each
with both an audio and video recording."* The notebook pairs **2,452**. Verified against the
Kaggle file listing: RAVDESS gives 1,440 speech (24 actors × 60 trials) + 1,012 song (23 × 44 —
actor 18 recorded no song). **Neither 2,452 nor 1,440 equals 1,360**, so the paper's own
accounting does not reconcile with the dataset's structure — but 2,452 is nearly double what the
paper describes, and it includes the **song** subset, where actors *sing* rather than speak and
where **disgust and surprised do not exist at all**.

**b) It does not implement Fig. 4.** Paper §VII.2 specifies r3d_18 → *"additional 3D convolutional
layers are stacked, followed by global average pooling"*. The notebook slices
`nn.Sequential(*list(backbone.children())[:-1])`, which **keeps r3d_18's own AdaptiveAvgPool3d**.
The volume is therefore already 1×1×1 before the `Conv3d(512,256,k=3,padding=1)` stack runs, so
each conv sees 26/27 zero-padding and degenerates to its centre tap — a stack of Linear layers
wearing Conv3d costumes. Cell 21's own comment shows the author knew.

**c) Its fused features are unscaled.** Cell 33 concatenates raw MFCC means (≈ −300) with 0–1
softmax probabilities, and `FusionMLP`'s first `BatchNorm` sits *after* the first `Linear`, so
nothing ever normalises the input.

---

## 3. What we changed, and what it cost or bought

| arm | change | video | fusion |
|---|---|---|---|
| paper (published) | — | 92.72 | **96.06** |
| notebook (committed) | — | 91.2 | 94.29 |
| **kinetics** | Kinetics channel stats + scaler | **81.97** ± 8.11 | **93.84** |
| **perclip** | per-clip norm restored + scaler | *running* | *running* |
| **refine** | + Fig. 4 on a real 2×7×7 volume | *running* | *running* |
| **speech** | + 1,440 clips, the paper's data scale | *queued* | *queued* |

### A negative result worth publishing

We replaced the paper's per-clip video standardization with **Kinetics channel statistics**, on the
theory that feeding Kinetics-pretrained weights per-clip-standardized inputs "fights" them. **The
theory is wrong.** Video fell **10.75 points** and became unstable — std 8.11, with folds
collapsing to 71.22 — and fusion landed *below* even the notebook's 94.29.

The likely mechanism runs the other way: per-clip standardization removes per-clip lighting and
contrast variation, and RAVDESS records every actor under consistent studio conditions, so
removing that nuisance variation matters more here than matching Kinetics' global statistics.

**This is a real finding**: "match the pretrained model's input statistics" is standard advice, and
on this dataset it costs 10 points.

### What the scaler is worth

Fusion fell only **0.45** below the notebook **while being fed video 10 points worse**. The
StandardScaler absorbed almost the entire hit. That is the headroom the per-clip arm should recover.

---

## 4. The depressed / not-depressed head

**RAVDESS contains no depression label.** It is 24 actors portraying 8 emotions on two fixed
sentences. `sad → depressed` builds a detector for *acted sadness*, which is not clinical
depression in either direction: acted sadness is not depression, and depression frequently presents
as blunted, flat affect rather than sad affect. Depression is measured over weeks (PHQ-8, BDI), not
in a 3-second clip.

**The measurement says the same thing, without needing the argument.** Post-hoc over the saved
out-of-fold predictions:

| mapping | positive | accuracy | majority baseline | **beats baseline by** | AUC |
|---|---|---|---|---|---|
| `sad → depressed` | 15.3% | **86.46** | 84.67 | **+1.79** | 81.58 |
| `sad + fearful` | 30.7% | 80.71 | 69.33 | +11.38 | 83.27 |
| **`negative_valence`** | 53.8% | **79.12** | 53.83 | **+25.29** | 86.68 |

**The depression mapping reports the highest accuracy and learns the least.** Because only 15.3% of
RAVDESS is sad, always answering "not depressed" scores **84.67%** — the model beats a hardcoded
constant by 1.79 points. `negative_valence` reports a *lower* number while doing **fourteen times**
more work. On **audio alone** the depression mapping scores **84.38% against an 84.67% baseline** —
*below* a constant classifier.

The valence head is real, not a class prior: per-emotion firing rates are angry .90, fearful .90,
disgust .88, sad .74 against neutral .15, calm .17, happy .27. (*Surprised* at .64 is the honest
wart — its valence is genuinely ambiguous.)

**Recommendation:** report `negative_valence` as the headline; keep the `depressed_*` rows beside
it **with their baselines in the same table**. A reader who sees 86.46 next to 84.67 cannot be
misled. A reader who sees 86.46 alone will be.

**Cut for the deadline:** fine-tuning on DAIC-WOZ's real PHQ-8 labels. It needs a signed access
agreement with days of lead time. The paper should report the transfer pathway as demonstrated and
clinical validation as pending data access.

---

## 5. Claims we withdrew

Kept visible so they are not cited from earlier drafts.

- **"The video key collision is an accuracy bug."** The collision is real and total — every one of
  2,452 video keys matches 2 files, confirmed against the Kaggle listing, because cell 7's `key`
  omits `modality`. But modality 01 (full-AV) and 02 (video-only) are the *same visual recording*,
  so `v.iloc[0]` picking either yields identical frames. **Cosmetic**: it buys determinism, not
  accuracy.
- **"Kinetics normalization is a fix."** Measured as a 10.75-point regression. See §3.
- **"The `refine` head is a bug."** It is a deliberate, documented deviation (cell 21's comment
  says so) — but it *does* mean the notebook doesn't implement the paper's Fig. 4. Reclassified
  from bug to unimplemented specification.
- **"The target is 94.29%."** That is the notebook's score. The paper's is 96.06%.

## 6. Known limitations

- **The metric is the paper's**: max-over-epochs accuracy on the very fold being reported.
  Optimistic by construction. Kept deliberately — changing it would make comparison to Table VII
  meaningless. You cannot claim to beat a number you measured differently.
- **Out-of-fold features use inner 2-fold**, not full nested CV — the affordable approximation.
  Training features come from models trained on half the training actors; test features from models
  retrained on all of them (standard stacking, as `sklearn`'s `StackingClassifier` does).
- **The paper does not state its epoch counts.** The notebook uses 15 for video, and its own fold-1
  log shows validation accuracy still rising at epoch 15 (90.02 → 91.24). Video is plausibly
  under-trained, and the paper's max-over-epochs metric rewards longer training — so a fair
  comparison would need an epoch count the paper never published.
- **The actor-independent numbers in §1 are a lower bound**: they were produced with the Kinetics
  regression in place. The per-clip re-run will raise them.
