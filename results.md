# Results

All numbers: macro-F1 on the same validation set (8,005 texts, group split), unless noted.
Public LB is computed on 30% of the test set only (7,501 rows), so small differences are noise.

## Runs
| Model | Train size | Epochs | Val macro-F1 | human F1 | ai F1 | fusion F1 | Public LB |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (C=30) | 32k | - | 0.8598 | 0.800 | 0.981 | 0.798 | - |
| Perplexity + stylometry, LightGBM | 32k | - | 0.7431 | 0.663 | 0.941 | 0.624 | - |
| XLM-R base, 512 tok | 32k | 3 | 0.904 | 0.840 | 0.990 | 0.882 | - |
| mDeBERTa-v3 base, 512 tok | 32k | 3 | 0.9079 | 0.853 | 0.994 | 0.876 | - |
| XLM-R base, 512 tok | 100k | 2 | 0.9221 | 0.869 | 0.995 | 0.903 | - |
| mDeBERTa-v3 base, 512 tok | 100k | 2 | 0.9301 | 0.883 | 0.996 | 0.912 | 0.93714 |
| **Blend** (mDeBERTa 100k 0.55, XLM-R 32k 0.42, XLM-R 100k 0.03; class multipliers human/ai/fusion = 1.0/0.8/0.8) | - | - | 0.9353 | 0.892 | 0.996 | 0.918 | 0.93864 |

Blend weights were searched on half A of the validation set (0.9379) and checked on half B (0.9304).

## Ablation (LightGBM, 32k train)
| Features | macro-F1 |
|---|---|
| perplexity only (10) | 0.5403 |
| stylometry only (10) | 0.7007 |
| perplexity + stylometry (20) | 0.7431 |

## Selected final submissions
- `submission_blend.csv` (LB 0.93864)
- `submission_enc_mdeberta-v3-base_ht_512_dom_100k.csv` (LB 0.93714)

## Accuracy by domain (single mDeBERTa, 100k)
| Domain | Accuracy |
|---|---|
| arxiv_papers | 0.974 |
| blog_authorship_corpus | 0.894 |
| bundestag | 0.926 |
| cnn_news | 0.968 |
| euro_court_cases | 0.879 |
| gutenberg | 0.960 |
| house_of_commons | 0.978 |
| student_essays | 0.972 |

## Kaggle submission history
| Public LB | Kaggle version | Val (того часу) | What it was |
|---|---|---|---|
| 0.71031 | 04 Version 1 | 0.8298 (val 1,001 texts) | best single model of the first pipeline: XLM-R, 256 tokens, random split |
| 0.86241 | 04 Version 2 | blend 0.9063 / best single 0.8715 (val 8,012, intermediate TF-IDF+SVD split) | blend of TF-IDF 0.48, GBM 0.04, XLM-R 256 tokens 0.48 [перевір: blend чи best single] |
| 0.91972 | 02 Version 2 | 0.9079 | single mDeBERTa-v3, 512 tokens, 32k train, 3 epochs |
| 0.92245 | 04 Version 3 | 0.9209 | blend of mDeBERTa 32k (0.46) and XLM-R 32k (0.54), val A/B 0.9185 / 0.9229 |
| 0.92554 | 04 Version 4 | 0.9251 | blend of two XLM-R (32k and 100k) |
| 0.92609 | 04 Version 5 | 0.9246 | same two XLM-R, different random weights |
| 0.93714 | 02 (mDeBERTa 100k) | 0.9301 | single mDeBERTa, 100k train |
| 0.93864 | 04 Version 6 | 0.9353 | final blend (mDeBERTa 100k, XLM-R 32k, XLM-R 100k) |


## Failed or excluded attempts
- Random split: validation 0.83 vs leaderboard 0.71 (versions of the same text in train and validation).
- Grouping via TF-IDF + SVD: giant groups, then almost all groups of size 1.
- Group split by shared 6-word phrases: works for human and fusion; ai texts have no verbatim overlap and stay ungrouped.
- mDeBERTa first run crashed: weights loaded in fp16 ("Attempting to unscale FP16 gradients"), fixed with `.float()`.
- First blend missed mDeBERTa (wrong files in the input dataset): val 0.9246, LB 0.92609.
- 256 -> 512 tokens together with the domain prefix: two changes at once, and the 256 run used the old split, so the effect cannot be attributed to either.
- Perplexity features alone: 0.54, stylometry alone: 0.70. Both far below TF-IDF (0.8598) and the encoders.
