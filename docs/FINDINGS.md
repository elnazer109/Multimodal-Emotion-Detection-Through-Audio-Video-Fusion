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

**a) It trains on the wrong data — and the extra data is the *easy* half.** Paper §VI.6: *"The
dataset has 1,360 distinct samples, each with both an audio and video recording."* The notebook
pairs **2,452**. Verified against the Kaggle file listing: RAVDESS gives 1,440 speech (24 actors ×
60 trials) + 1,012 song (23 × 44 — actor 18 recorded no song). **Neither 2,452 nor 1,440 equals
1,360**, so the paper's own accounting does not reconcile with the dataset's structure.

The notebook's extra ~1,000 clips are the **song** subset, where actors *sing* rather than speak
and where **disgust and surprised do not exist at all**. Measured on the held-out actor
predictions, song is far *easier*:

| model | speech | song | |
|---|---|---|---|
| audio | 45.14 | **69.27** | song **+24.13** |
| fusion | 43.19 | **59.78** | song **+16.59** |
| video | 46.74 | 42.69 | speech +4.05 |

Neutral alone runs 25.0% on speech against **84.8%** on song. Singing exaggerates emotional
prosody — pitch, duration and intensity all become more distinct — which is why *audio* gains 24
points while video is unaffected.

So the notebook is, if anything, **flattered** by training on ~80% more data than the paper
describes, most of it easier — and it *still* lands 1.77 points below the paper's number.

(We predicted the opposite and were wrong; see §5.)

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

| arm | change | audio | video | fusion |
|---|---|---|---|---|
| paper (published) | — | 79.24 | **92.72** ± 3.05 | **96.06** |
| notebook (committed) | — | 80.67 | **83.24** | 94.29 |
| **kinetics** | Kinetics channel stats + scaler | 81.81 | 81.97 ± 8.11 | 93.84 |
| **perclip** | per-clip norm restored + scaler | 81.73 | 78.25 ± 10.88 | 94.25 |
| **refine** | + Fig. 4 on a real 2×7×7 volume | *running* | *running* | *running* |
| **epochs** | + video 15 → 35 epochs | *queued* | *queued* | *queued* |

### The finding: the paper's video number is not reproducible

**No reproduction reaches it — including the notebook's own.**

| | video mean | sd | fold range |
|---|---|---|---|
| paper, Table VII | **92.68** | **3.05** | 88.2 – 96.6 |
| notebook | 83.24 | — | (fold 1 = 91.24, its best) |
| kinetics | 81.97 | 8.11 | 71.2 – 88.8 |
| per-clip | 78.25 | 10.88 | 65.7 – 93.5 |

Two things stand out. The paper's video sits **9–14 points above** every independent run of the
same architecture. And its **fold-to-fold spread is sd 3.05** where every reproduction shows
**8–11** — the published result is not just higher, it is far more stable than the architecture
appears able to be.

Our audio, by contrast, **matches and slightly exceeds** the paper (81.7–81.8 vs 79.24), so the
pipeline is not broadly broken. The discrepancy is specific to video.

### Retracted: "Kinetics normalization is a 10.75-point regression"

**This claim was wrong and is withdrawn.** It came from comparing our video against the *paper's*
92.72 and against the notebook's *fold 1* (91.24 — its single best fold), rather than against the
notebook's actual mean of **83.24**.

Measured correctly, against the notebook's mean:

| arm | video vs notebook | fusion vs notebook |
|---|---|---|
| kinetics | −1.27 | −0.45 |
| per-clip | −4.99 | −0.04 |

And a **paired t-test across identical folds** (same seed, same splits) says the two arms are
indistinguishable:

```
video   perclip − kinetics = −3.72   t = −0.77   p = 0.486   not significant
fusion  perclip − kinetics = +0.41   t = +0.39   p = 0.719   not significant
audio   perclip − kinetics = −0.08   t = −0.13   p = 0.903   not significant
```

**The normalization choice does not measurably matter here.** With video sd of 8–11 across only 5
folds, the standard error is ~4–5 points — large enough to swallow both changes whole. An entire
narrative was built on noise; it is retained here as §5 material rather than deleted.

### What the StandardScaler is worth: nothing measurable

Fusion: notebook 94.29, per-clip+scaler **94.25** (−0.04), kinetics+scaler 93.84 (−0.45). The
scaler was the one change with a clean mechanical rationale — raw MFCC means near −300 beside 0–1
softmax probabilities, with `FusionMLP`'s first BatchNorm sitting *after* the first Linear — and it
moves nothing. The fusion MLP evidently copes with the scale mismatch on its own.

**Fusion is also strikingly insensitive to video quality**: video ranging 78.25 → 81.97 across arms
moves fusion by 0.41. The audio branch and the softmax probabilities appear to carry the fusion
result, with the 512-dim video embedding contributing little.

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
- **"Kinetics normalization is a fix."** Withdrawn — no measurable effect either way (§3).
- **"Kinetics normalization is a 10.75-point regression."** Also withdrawn, and the more
  instructive error. Having found the first claim wrong, I over-corrected into an equally confident
  claim in the opposite direction — built from comparing against the paper's 92.72 and the
  notebook's single best fold rather than its mean. A paired t-test says p=0.486. **Being wrong
  twice in opposite directions about the same 20 lines of code is the signature of reading noise as
  signal**, and the fix was not a better theory but a baseline (the notebook's actual mean) and a
  significance test.
- **"Song is a contaminating domain; dropping it will raise accuracy."** Backwards. Song is the
  *easier* subset by 16–24 points (§2a). Dropping it would delete the easy half and lower the
  score. The arm was built and then cancelled before it consumed a GPU slot, on the strength of the
  saved predictions.
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
