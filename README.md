# AI-Generated Text Detection · HW3 · UCU AI 2026

Three-class text classification (**human / ai / fusion**) for English and German texts.
Competition: Kaggle "AI-Generated Text Detection | HW3 | UCU AI 2026". Metric: macro-F1.

Author: Bohdana Zubrytska | Kaggle username: @bohdanazubrytska

## Final results
| Submission | Public LB | Val macro-F1 |
|---|---|---|
| Blend (mDeBERTa 100k + XLM-R 32k + XLM-R 100k, class thresholds) | 0.93864 | 0.9353 |
| Single mDeBERTa-v3-base (100k train) | 0.93714 | 0.9301 |

Both files are selected as final submissions on Kaggle.

All numbers, per-class F1 and the full submission history are in [`results.md`](results.md).

## Repository structure
| Path | Runs on | What it does |
|---|---|---|
| `notebooks/00_group_split.ipynb` | CPU | Builds a group-aware train/validation split (versions of one text stay on one side). Writes `split.csv` (train ~100k) and `split_32k.csv` (train ~32k), same 8,005 validation texts. |
| `notebooks/01_eda_baseline_tfidf.ipynb` | CPU | EDA (class balance by language/domain, text lengths, examples), TF-IDF word + char n-grams with LogReg / LinearSVC, confusion matrix, top words per class. |
| `notebooks/02_encoder_finetune.ipynb` | GPU T4 | Fine-tunes XLM-R base or mDeBERTa-v3 base, 512 tokens (first + last half for long texts), `[domain \| language]` prefix. Run four times with different `MODEL` / `SPLIT_FILE` / `EPOCHS` (table at the top of the notebook). |
| `notebooks/03_perplexity_stylometry_shap.ipynb` | GPU T4 | Perplexity features from Qwen2.5-0.5B + stylometry, LightGBM, ablation, SHAP. |
| `notebooks/04_blend_final.ipynb` | CPU | Weighted blend of encoder probabilities + class multipliers tuned for macro-F1, with an A/B check. |
| `results.md` | - | All runs, per-class F1, ablation, Kaggle submission history. |
| `report/report.pdf` | - | Technical report (1-2 pages). |
| `submissions/` | - | The two selected final submission files. |
| `probs/` | - | Validation and test probabilities of the encoder runs (`*_val.npy`, `*_test.npy`), so `04` runs in a minute without GPU. |

The competition data is not included (course-use license).

## How to reproduce (Kaggle)
1. Create a notebook from the competition page, so the data is attached.
2. Run `00` first. Attach its output as input to `01`, `02`, `03`.
3. `01` and `03` read `split_32k.csv`. `02` reads `split.csv` by default; set `SPLIT_FILE`, `MODEL`, `EPOCHS` as in the table at the top of the notebook for the other runs (about 1 to 4 hours of T4 each).
4. Save the `*_val.npy` / `*_test.npy` files of the `02` runs as a dataset, attach it to `04` and run it.

Reproducibility notes:
- The split, sampling, LightGBM and the blend search use seed 42.
- Encoder fine-tuning does not fix a torch seed, so a rerun gives slightly different numbers.
- Blend weights are found by random search and depend on the order of input files, so reruns can give a slightly different blend (we saw 0.9246 vs 0.9251 on the same models).
- Weights of pretrained models are downloaded from Hugging Face (Internet On in Kaggle).

## Validation
The dataset contains a human text together with its AI and fusion versions, and these always lie on the same side of the train/test split. A random split of `train.csv` leaked versions of one text into validation: validation macro-F1 was 0.83 but the leaderboard gave 0.71. We therefore split by groups: texts that share many 6-word phrases are merged into one group (fusion is a partly rewritten human text), and whole groups go to train or validation. Limitation: AI texts have no verbatim overlap with the human text, so they stay ungrouped; validation for the `ai` class may be slightly optimistic. After this change, validation and the public leaderboard agree within about 0.012.

## Experiments
| # | Experiment | Reference | Result (val macro-F1) |
|---|---|---|---|
| 1 | TF-IDF (word + char) + LogReg, 32k | - | 0.8598 |
| 2 | XLM-R base, 512 tokens + domain prefix, 32k | TF-IDF | 0.904 |
| 3 | mDeBERTa-v3 instead of XLM-R, 32k | XLM-R 32k | 0.9079 |
| 4 | More data (100k instead of 32k; 2 instead of 3 epochs) | 32k runs | XLM-R 0.9221, mDeBERTa 0.9301 |
| 5 | Blend + class thresholds | best single | 0.9353 |
| 6 | Perplexity + stylometry, LightGBM, 32k | TF-IDF / XLM-R, 32k | 0.7431 (perplexity only 0.5403, stylometry only 0.7007) |

Failed or inconclusive attempts: random split (val 0.83 vs LB 0.71); two group-split attempts that produced giant groups or only singletons; mDeBERTa crashed in fp16 on the first run; the first blend missed mDeBERTa because of wrong input files (LB 0.92609); 256 → 512 tokens changed together with the domain prefix, so their effects cannot be separated.

## Explainability
TF-IDF + LogReg top words: human texts are pulled by placeholder-like tokens and student-essay markers (`student_name`, `urllink`, `conclusion`); ai texts by informal words (`let`, `maybe`, `seriously`, `super`); fusion texts by connectors (`furthermore`, `moreover`, `additionally`). SHAP on the LightGBM model: [впиши: які ознаки найважливіші на графіку з ноутбука 03 і твій висновок, 1-2 речення].

## Rules compliance
- No proprietary LLMs were used to label, generate or extract features. Open-weight models only: XLM-R, mDeBERTa-v3, Qwen2.5-0.5B.
- Trained on `train.csv` only. Vectorizers and models were fitted on the train part only; validation was used for model selection and for the blend weights and thresholds. Test rows were never used for fitting.
- AI tools were used to help write code. The report is written without AI.

## References
- Schaaf, Bönisch, Mehler. GhostWriter dataset, LREC 2026.
- Conneau et al. XLM-R. · He et al. DeBERTaV3 (mDeBERTa). · Qwen2.5.
- Lundberg & Lee. SHAP. · Ke et al. LightGBM.
