# Multimodal Emotion Detection Through Audio-Video Fusion

Implementation and reproduction study of ***Emotion Unlocked: Multimodal Emotion Detection Through
Audio-Video Fusion*** (Vasuki, Padhma Priya M, Pooja V — SSN College of Engineering, preprint
[SSRN 5274911](https://ssrn.com/abstract=5274911)), on the
[RAVDESS](https://www.kaggle.com/datasets/orvile/ravdess-dataset) audio-visual dataset.

The system classifies **8 emotions** — neutral, calm, happy, sad, angry, fearful, disgust,
surprised — from three-second clips, by fusing what it hears with what it sees.

---

## How it works

Three models, trained in sequence:

| stage | input | architecture | output |
|---|---|---|---|
| **Audio** | 40-dim MFCCs (22.05 kHz; pre-emphasis → STFT → Mel → DCT) | CNN(64→512) → BiLSTM(256) → BiLSTM(512) → BiRNN(512) → Dense | 8-class softmax |
| **Video** | 16 face crops @ 112×112 sampled across the clip | R3D-18 (Kinetics-pretrained) → stacked 3D convs → global avg pool | 8-class softmax + 512-d embedding |
| **Fusion** | **568-d** = 40 mean-MFCC + 8 audio softmax + 512 video embedding + 8 video softmax | MLP(256→128→64→32) | 8-class softmax |

The premise: audio carries prosody (tone, pitch, rhythm), video carries facial expression, and the
two are complementary — so a model that sees *and* hears beats either alone. It does, by a wide
margin.

**Data:** 2,452 paired clips — 24 actors × 2 fixed sentences × 2 repetitions, across speech (1,440)
and song (1,012).

---

## Results

Under the paper's own protocol, so every row compares directly to its Table VII:

| model | paper | this repo |
|---|---|---|
| Audio only | 79.24% | **80.42%** |
| Video only | 92.72% | **94.58%** |
| **Fusion** | **96.06%** | **96.33%** |

Fusion beats both single modalities decisively, confirming the paper's central claim.

> **Honest note:** 96.33% vs 96.06% is a *statistical tie* (p = 0.687 over 5 folds). We **reproduce**
> the published result — we do not beat it.

---

## What the reproduction found

Full write-up: **[`docs/FINDINGS.md`](docs/FINDINGS.md)** · rendered:
**[`docs/FINDINGS.pdf`](docs/FINDINGS.pdf)**

### 1. One character cost 9.4 points of video accuracy

The paper (§VII.2) specifies R3D-18 → *stacked 3D convolutions* → global average pooling. The
notebook wrote:

```python
self.backbone = nn.Sequential(*list(backbone.children())[:-1])   # keeps R3D-18's OWN avgpool
```

`[:-1]` drops only the final `fc`, **keeping R3D-18's own `AdaptiveAvgPool3d`**. The feature volume
is therefore already 1×1×1 when the 3D convolutions run — each sees 26/27 zero-padding and collapses
to a single tap. **The paper's Fig. 4 was never actually built.**

`[:-2]` keeps `stem…layer4`, giving the convolutions a real (B, 512, 2, 7, 7) volume:

| | video |
|---|---|
| before (pools first) | 83.24% |
| **after (Fig. 4 built)** | **92.62%** |
| paper reports | 92.72% |

**+14.37 points**, paired *t*-test p = 0.039 — landing within **0.10** of the paper. The fix costs
nothing (~0.35 GMAC/sample against R3D-18's own ~40 GFLOP). **The paper was right; the
implementation was not the paper.**

### 2. The benchmark substantially measures actor identity

Five-fold cross-validation splits **samples**, not **actors**. Every actor recorded each sentence
twice — so repetition 1 trains while repetition 2 validates: same face, same voice, same sentence,
same emotion. The model can score well by recognising *who is speaking*.

Re-run with actors held out (`GroupKFold`):

| protocol | fusion |
|---|---|
| standard 5-fold (actors on both sides) | **96.06%** |
| **actor-independent** (5 actors held out) | **≈ 63%** |

The model does still read emotion — errors cluster affectively (sad↔fearful, angry↔disgust), not
randomly — but on **people it has never seen** it is far weaker than the headline implies. Anyone
deploying this should expect the lower number.

Two independent runs agree within ~1 point, so this does not depend on any tuning choice.

---

## Repository layout

| path | what it is |
|---|---|
| **`Paper_worK/paper-notebook-corrected.ipynb`** | **The corrected pipeline** — Fig. 4 built, actor-independent switch, scaled fusion input. Loads saved results at the bottom, so **you can read the numbers without running it**. |
| `Paper_worK/paper-notebook.ipynb` | The original implementation, **kept unmodified** as the reproduction baseline. |
| `Paper_worK/video paper.pdf` | The paper being implemented. |
| `docs/FINDINGS.md` / `.pdf` | Full write-up. §5 lists every claim that did **not** survive measurement. |
| `docs/design-decisions.md` | Why each choice was made, and what was cut. |
| `src/` | The reproduction experiments — one file per configuration. |
| `results/` | Every CSV and log the runs produced. Nothing here is hand-typed. |
| `results.ipynb` | Presentation only — loads the CSVs, runs in seconds. |

### `src/` at a glance

| file | role |
|---|---|
| `cache_faces.py` | One-time face extraction (2,452 clips → 16 crops each). ~1h CPU, never repeated. |
| `paper_protocol.py` | Kinetics-normalisation arm — kept as a **negative result**. |
| `paper_perclip.py` | Per-clip normalisation (the paper's own choice) + scaled fusion input. |
| **`paper_refine.py`** | **The fix** — the paper's Fig. 4, as specified. |
| `paper_epochs.py` | As above with 35 video epochs (reported with a caveat). |
| `train_actor_independent.py` | The actor-independent evaluation. |
| `analyze_binary_mapping.py` | Collapses 8 emotions to a binary affect head, post-hoc. |
| `patch_notebook.py` | Generates the corrected notebook from the original. |
| `make_report.py` | One command: pull results → rebuild tables → re-render the PDF. |

---

## Running it

The dataset is **25.6 GB** and training needs a GPU, so everything runs on Kaggle.

```bash
pip install -r requirements.txt

# 1. build the face cache once (CPU, ~1h)
kaggle kernels push -p kaggle/cache

# 2. run an experiment (GPU, ~4h)
kaggle kernels push -p kaggle/refine --accelerator NvidiaTeslaT4

# 3. pull results, rebuild every table, re-render the PDF
python src/make_report.py
```

> **`--accelerator NvidiaTeslaT4` is required.** Kaggle's default GPU is a P100 (sm_60) while the
> preinstalled PyTorch supports sm_70+ only — so `torch.cuda.is_available()` returns `True`, every
> CUDA launch fails, and the run silently falls back to CPU. It looks healthy for an hour and takes
> days. The scripts here refuse to start rather than degrade quietly.

## Credits

- **Paper:** Vasuki, P., Padhma Priya, M., Pooja, V. — SSN College of Engineering (SSRN 5274911)
- **Dataset:** Livingstone & Russo (2018), *RAVDESS*, PLoS ONE 13(5): e0196391 — CC BY-NC-SA 4.0
