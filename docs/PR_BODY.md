# Actor-independent re-evaluation, reproduction fixes, and a binary affect head

> Draft PR body. **Not opened** — opening it notifies the repo owner and the team, which is
> Ehab's call, not an automated one. Numbers marked *pending* fill in as the last runs land.

## TL;DR

The published **96.06%** is substantially a measurement of **actor identity**. Under
actor-independent evaluation the same architecture scores **62.98%**. A second, independent leak in
the fusion features costs a further **12.73** points. This PR measures both, fixes what can be
fixed, and reports the honest numbers alongside the paper-comparable ones.

It also finds that this repo's notebook **never reproduced its own paper** (94.29 vs 96.06), and
why.

## 1. The leak is in the publication

Paper §F, *Cross-Validation Strategy*:

> "five-fold cross-validation is adopted. The database is split into five subsets wherein: Four
> subsets are used for training. One subset is used for validation."

Subsets **of samples**. Actors and speakers are never mentioned in the paper. RAVDESS is 24 actors
delivering **two fixed sentences**, each recorded **twice** — so repetition 1 trains while
repetition 2 validates: same face, same voice, same sentence, same emotion, same intensity.

| protocol | fusion |
|---|---|
| paper, Table VII | **96.06** |
| actor-independent (GroupKFold, 5 actors held out per fold) | **62.98** ± 7.96 |
| + out-of-fold fusion features | **50.25** ± 9.10 |

Errors cluster affectively (sad↔fearful 87, angry↔disgust 82, happy↔calm 79), so the architecture
does read emotion — just far less than advertised.

## 2. The notebook is not the paper

| | audio | video | fusion |
|---|---|---|---|
| paper Table VII | 79.24 | 92.72 | **96.06** |
| notebook as committed | ~80.7 | 91.2 | **94.29** |

- **Wrong data.** Paper §VI.6: *"1,360 distinct samples."* The notebook pairs **2,452** — including
  the **song** subset, where actors sing rather than speak and where **disgust and surprised do not
  exist**. (Verified: speech 1,440 = 24×60; song 1,012 = 23×44, actor 18 recorded no song. Neither
  equals 1,360; the paper's accounting does not reconcile.)
- **Fig. 4 not implemented.** §VII.2 specifies r3d_18 → stacked 3D convs → global average pooling.
  The notebook slices `children()[:-1]`, keeping r3d_18's own avgpool, so the volume is 1×1×1
  before the convs run and each degenerates to its centre tap.
- **Unscaled fusion input.** Raw MFCC means (≈ −300) concatenated with 0–1 softmax probabilities;
  `FusionMLP`'s first BatchNorm sits *after* the first Linear.

## 3. Arms (all on the paper's own protocol → comparable to 96.06)

| arm | change | fusion |
|---|---|---|
| kinetics | Kinetics channel stats + scaler | **93.84** — regression |
| perclip | per-clip norm + scaler | *pending* |
| refine | + Fig. 4 on a real 2×7×7 volume | *pending* |
| speech | + 1,440 clips, the paper's data scale | *pending* |

### Negative result worth keeping

Replacing per-clip standardization with Kinetics channel statistics — standard advice for
pretrained backbones — **cost 10.75 points of video** (81.97 ± 8.11 vs the paper's 92.72, folds
collapsing to 71.22). RAVDESS records each actor under consistent studio lighting, so per-clip
standardization removes nuisance variation that matters more here than matching global statistics.

## 4. Binary affect head

Post-hoc over saved out-of-fold predictions — no retraining, so every mapping is reportable from
one run.

| mapping | accuracy | majority baseline | **beats baseline by** |
|---|---|---|---|
| `sad → depressed` | **86.46** | 84.67 | **+1.79** |
| `sad + fearful` | 80.71 | 69.33 | +11.38 |
| **`negative_valence`** | **79.12** | 53.83 | **+25.29** |

**The depression mapping scores highest and learns least.** Only 15.3% of RAVDESS is sad, so
answering "not depressed" always scores 84.67. On **audio alone** it scores 84.38 against an 84.67
baseline — *below a constant classifier*.

RAVDESS has **no depression label**; `sad → depressed` detects *acted sadness*, which is not
clinical depression in either direction. Recommend `negative_valence` as the headline with the
`depressed_*` rows beside it **and their baselines in the same table**.

DAIC-WOZ fine-tuning (real PHQ-8 labels) was cut — it needs a signed access agreement with days of
lead time.

## Layout

| path | |
|---|---|
| `docs/FINDINGS.md` | full write-up; §5 lists claims withdrawn after checking |
| `docs/2026-07-16-actor-independent-evaluation.md` | design + decisions log |
| `src/cache_faces.py` | one-time uint8 face cache (CPU) |
| `src/paper_*.py` | the paper-protocol arms |
| `src/train_actor_independent.py` | the honest protocol |
| `src/analyze_binary_mapping.py` | post-hoc binary head |
| `results.ipynb` | loads the CSVs; re-runs in seconds |

## Reviewer notes

- The **metric is the paper's** (max-over-epochs on the scored fold). Optimistic by construction,
  kept deliberately — you cannot claim to beat a number you measured differently.
- Out-of-fold features use **inner 2-fold**, not full nested CV — the affordable approximation.
- The paper **does not state epoch counts**; the notebook uses 15 for video and its own log shows
  validation still rising at 15, so video is plausibly under-trained.
- Video pretrained weights are **sha256-verified** against the official `r3d_18-b3b3357e.pth`
  before training, because Kaggle kernels have no network here and the checkpoint comes from a
  third-party dataset mirror.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
