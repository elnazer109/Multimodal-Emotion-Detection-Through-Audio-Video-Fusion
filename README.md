# Multimodal Emotion Detection Through Audio-Video Fusion

Reproduction and improvement of ***Emotion Unlocked: Multimodal Emotion Detection Through
Audio-Video Fusion*** (Vasuki, Padhma Priya M, Pooja V — SSN College of Engineering, preprint
[SSRN 5274911](https://ssrn.com/abstract=5274911)), on the
[RAVDESS](https://www.kaggle.com/datasets/orvile/ravdess-dataset) dataset — classifying **8
emotions** from three-second clips by fusing audio with video.

---

## 🎯 Result: accuracy improved 94.29% → 96.45%

The starting notebook scored **94.29%** fusion accuracy. After correcting the video model, this
version scores **96.45%** — a **+2.16 point** gain on the same 5-fold protocol.

| model | starting notebook | **corrected** | gain |
|---|---|---|---|
| Audio | 80.67% | 81.77% | +1.10 |
| **Video** | 83.24% | **95.64%** | **+12.40** |
| **Fusion** | **94.29%** | **96.45%** | **+2.16** |

### What made the difference

The starting notebook never implemented the paper's **Fig. 4** video architecture. It sliced the
R3D-18 backbone like this:

```python
self.backbone = nn.Sequential(*list(backbone.children())[:-1])   # keeps R3D-18's OWN avg-pool
```

`[:-1]` removes only the final classification layer but **keeps R3D-18's own average-pooling** — so
the feature map was already collapsed to 1×1×1 before the 3D convolution layers ran. Those
convolutions were processing nothing but padding. The paper's Fig. 4 was never actually built.

Changing one character — `[:-2]` — keeps the spatiotemporal feature map (2×7×7) so the convolutions
do real work:

| video model | accuracy |
|---|---|
| before (feature map collapsed) | 83.24% |
| **after (Fig. 4 built correctly)** | **95.64%** |

That **+12 point** jump in the video model is statistically significant (paired *t*-test,
p = 0.039) and it carried the fusion model to 96.45%. The fix costs essentially no extra compute.

> **On the published paper:** the paper reports 96.06%. Our 96.45% is a *statistical tie* with it
> (p = 0.687), not a claim of beating it — the improvement here is over the **starting
> implementation**, which is what was asked. See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full
> statistical treatment.

---

## How it works

Three models, trained in sequence, then fused:

| stage | input | architecture | output |
|---|---|---|---|
| **Audio** | 40-dim MFCCs (pre-emphasis → STFT → Mel → DCT) | CNN(64→512) → BiLSTM(256) → BiLSTM(512) → BiRNN(512) → Dense | 8-class softmax |
| **Video** | 16 face crops @ 112×112 across the clip | R3D-18 (Kinetics-pretrained) → 3D convs → global avg-pool | 8-class softmax + 512-d embedding |
| **Fusion** | **568-d** = 40 mean-MFCC + 8 audio-softmax + 512 video-embedding + 8 video-softmax | MLP(256→128→64→32) | 8-class softmax |

Audio carries prosody (tone, pitch, rhythm); video carries facial expression. They are
complementary, so fusion beats either alone — audio 80%, video 95%, **fusion 96%**.

**Data:** 2,452 paired clips — 24 actors × 2 sentences × 2 repetitions, across speech (1,440) and
song (1,012).

---

## Deeper analysis

The full write-up is in **[`docs/FINDINGS.md`](docs/FINDINGS.md)**
(rendered: **[`docs/FINDINGS.pdf`](docs/FINDINGS.pdf)**). Two findings beyond the accuracy result:

**How robust is 96% really?** The standard 5-fold split places each actor in both training and
validation (they record every sentence twice). Re-evaluated with **actors fully held out**
(`GroupKFold`), the same model scores **≈ 69%** — a ~27-point drop. It recognises emotion in *seen*
speakers far better than in new ones. Worth knowing before deployment. This is a property of the
dataset, not of this implementation.

**A depressed / not-depressed head** was explored as a post-hoc mapping of the 8 emotions. It is
reported with each mapping's majority-class baseline beside it — because RAVDESS is imbalanced,
"always predict not-depressed" already scores 84.67%, so raw accuracy there is misleading. Details
in the findings.

---

## Repository layout

| path | what it is |
|---|---|
| **`Paper_worK/paper-notebook-corrected.ipynb`** | **The corrected pipeline, executed end-to-end on Kaggle (T4 GPU).** Every cell has real output; opens with the accuracy result and reproduces the Table VII comparison from the runs. |
| `Paper_worK/paper-notebook.ipynb` | The original, kept unmodified as the baseline. |
| `Paper_worK/video paper.pdf` | The paper being implemented. |
| `docs/FINDINGS.md` / `.pdf` | Full write-up, with statistics. §5 lists claims that did *not* survive testing. |
| `docs/design-decisions.md` | Why each choice was made. |
| `src/` | One script per experiment (see below). |
| `results/` | Every CSV and log the runs produced — nothing hand-typed. |
| `results.ipynb` | Presentation only; loads the CSVs, runs in seconds. |

**`src/` key files:** `cache_faces.py` (one-time face extraction) · **`paper_refine.py`** (the Fig. 4
fix) · `train_actor_independent.py` (the held-out-actors evaluation) · `analyze_binary_mapping.py`
(binary head) · `make_report.py` (one command: pull results → rebuild tables → re-render the PDF).

---

## Running it

Dataset is 25.6 GB and training needs a GPU, so everything runs on Kaggle.

```bash
pip install -r requirements.txt
# edit kaggle/*/kernel-metadata.json -> put your Kaggle username in the "id" fields, then:
kaggle kernels push -p kaggle/cache                                # face cache (CPU, ~1h)
kaggle kernels push -p kaggle/refine --accelerator NvidiaTeslaT4   # train    (GPU, ~4h)
python src/make_report.py                                          # pull + rebuild the report
```

> **`--accelerator NvidiaTeslaT4` is required.** Kaggle's default GPU (P100) is incompatible with
> the preinstalled PyTorch, so a run silently falls back to CPU and takes days. The scripts refuse
> to start rather than degrade quietly.

## Credits

- **Paper:** Vasuki, P., Padhma Priya, M., Pooja, V. — SSN College of Engineering (SSRN 5274911)
- **Dataset:** Livingstone & Russo (2018), *RAVDESS*, PLoS ONE 13(5): e0196391 — CC BY-NC-SA 4.0
