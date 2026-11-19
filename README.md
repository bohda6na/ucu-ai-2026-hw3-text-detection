# AI-Generated Text Detection · HW3 · UCU AI 2026

Three-class text classification (**human / ai / fusion**) for English and German texts.
Competition: Kaggle "AI-Generated Text Detection | HW3 | UCU AI 2026". Metric: macro-F1.

Author: [ім'я, прізвище] · Kaggle username: [нік]

## Final results
| Submission | Public LB | Val macro-F1 |
|---|---|---|
| Blend (mDeBERTa + 2×XLM-R, class thresholds) | 0.93864 | 0.9353 |
| Single mDeBERTa-v3-base (100k train) | 0.93714 | 0.9301 |

Both files are selected as final submissions on Kaggle.

## Repository structure
| Notebook | Runs on | What it does |
|---|---|---|
| `notebooks/00_group_split.ipynb` | CPU | [1-2 речі: груповий спліт, чому, що виходить] |
| `notebooks/01_eda_baseline_tfidf.ipynb` | CPU | [EDA, TF-IDF + LogReg/SVM] |
| `notebooks/02_encoder_finetune.ipynb` | GPU T4 | [XLM-R / mDeBERTa, 512 токенів, домен у тексті] |
| `notebooks/03_perplexity_stylometry_shap.ipynb` | GPU T4 | [Qwen-перплексія, стилометрія, GBM, SHAP] |
| `notebooks/04_blend_final.ipynb` | CPU | [ансамбль і пороги] |
| `report/report.pdf` | - | Technical report (1-2 pages) |

## How to reproduce (Kaggle)
1. Create a notebook from the competition page, so the data is attached.
2. Run `00` first. It writes `split.csv` and `split_32k.csv` (same 8,005 validation texts).
3. Attach the output of `00` as input to `01`, `02`, `03`, then run them.
   `02` is run several times by changing `MODEL`, `SPLIT_FILE`, `EPOCHS` (see the table at the top of the notebook).
4. Save the `*_val.npy` / `*_test.npy` files of the `02` runs as a dataset, attach them to `04`, run it.
Seeds: 42 everywhere. GPU: Kaggle T4.

## Validation
[Своїми словами: чому груповий спліт, що не групується (ai), що це означає для валідації.]

## Experiments
| # | Experiment | Reference | Result |
|---|---|---|---|
| 1 | TF-IDF (word+char) + LogReg | - | 0.8598 |
| 2 | XLM-R, 512 tokens, 32k train | TF-IDF | 0.904 |
| 3 | mDeBERTa instead of XLM-R, 32k | XLM-R 32k | 0.9079 |
| 4 | More data (100k) | 32k | XLM-R 0.9221, mDeBERTa 0.9301 |
| 5 | Blend + class thresholds | best single | 0.9353 |
| 6 | Perplexity + stylometry (GBM) | TF-IDF / XLM-R, 32k | [впиши після перезапуску 03] |

Failed attempts: [випадковий спліт (val 0.83 → LB 0.71), два невдалі групові спліти, fp16-помилка mDeBERTa, ансамбль без mDeBERTa.]

## Explainability
[2-3 речення: що показали слова TF-IDF і SHAP, свій висновок.]

## Rules compliance
- No proprietary LLMs were used to label, generate or extract features. Open-weight models only: XLM-R, mDeBERTa-v3, Qwen2.5-0.5B.
- Trained only on `train.csv`. Test rows were never used for fitting (vectorizers fitted on train only).
- AI tools were used to help write code. The report is written without AI. [перевір це речення, воно має бути правдою]

## References
- Schaaf, Bönisch, Mehler. GhostWriter dataset, LREC 2026.
- Conneau et al. XLM-R. · He et al. DeBERTaV3 (mDeBERTa). · Qwen2.5.
- Lundberg & Lee. SHAP. · Ke et al. LightGBM.
- [додай ще, якщо брала код чи ідеї звідкись]
