# TransferIQ: Intelligent Athlete Market Value Prediction Engine Using Integrated DataStreams |

Regression pipeline that predicts a player's market value (EUR millions) from performance statistics, injury history, social-media metrics and contract data. Built during the Infosys Springboard AI/ML virtual internship (Dec 2025 - Jan 2026) and revised in Nov 2026 with a stricter evaluation (see *Revision history*).

## Results

450 players, 33 features after cleaning. 90 players (20%) are held out as a test set and used once; models are tuned and compared with 5-fold CV on the other 360. Market value ranges from EUR 1.2M to EUR 61.3M (mean about EUR 16.6M). MAE and RMSE are in EUR millions, after converting predictions back from the log scale.

| Model | CV R² (mean ± std) | CV MAE | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| Mean baseline | -0.072 ± 0.058 | 6.83 | -0.100 | 8.60 | 11.67 |
| Ridge (linear baseline) | 0.903 ± 0.052 | 2.08 | 0.955 | 1.83 | 2.37 |
| Random Forest | 0.843 ± 0.047 | 2.53 | 0.908 | 2.27 | 3.38 |
| XGBoost | 0.873 ± 0.049 | 2.24 | 0.953 | 1.71 | 2.40 |
| LightGBM | 0.887 ± 0.056 | 2.13 | 0.954 | 1.72 | 2.40 |

- The test set is small, so test numbers are uncertain: for the selected model the bootstrap 95% interval is R² 0.94-0.97 and MAE EUR 1.52-2.14M. CV is the more reliable estimate.
- Ridge, XGBoost and LightGBM perform about the same (their CV differences are smaller than the fold-to-fold spread). Random Forest is clearly behind. The model was selected by lowest CV MAE (Ridge); the boosted models are a close second.
- A typical error of about EUR 2M is roughly 12% of the average player value.

![Model comparison](results/model_comparison.png)
![Actual vs predicted](results/actual_vs_predicted.png)

## What drives the predictions?

Permutation importance on the test set puts performance statistics on top: `defensive_index`, `goals_per_match`, `tackles`, `goals`, `matches_played`.

![Feature importance](results/feature_importance.png)

**Feature-group ablation** (CV R² when a group is removed):

| Features used | Ridge | LightGBM |
|---|---|---|
| all features | 0.903 | 0.887 |
| without social | 0.903 | 0.889 |
| without injury | 0.900 | 0.887 |
| without contract | 0.902 | 0.887 |

Removing social-media, injury or contract features does not change performance for either model. On this dataset there is **no evidence** that social sentiment or injury history improve the valuation; the earlier claim that they are "significant drivers" was not supported and has been removed.

## Method

1. **Cleaning:** drop identifiers and the text trend label `market_value_`; fill missing `injury_prone_body_parts` (97 rows) with "None".
2. **Leakage control:** features with |correlation| > 0.9 with the target are dropped (`performance_score` 0.98, `estimated_resale_value` 0.94). Computed on the training portion only.
3. **Duplicate columns:** `discipline_index` and `injury_risk_assessment` are exact copies of `injury_risk_score` (correlation 1.0), and `years_remaining_contract` duplicates `contract_end_year`; one of each is kept.
4. **Preprocessing (inside the CV pipeline, so nothing leaks):** one-hot encoding for categoricals, standard scaling for numerics, `log1p` on the target.
5. **Models:** mean baseline, Ridge, Random Forest, XGBoost, LightGBM, each tuned with a small grid search under 5-fold CV. No early stopping on the test set.
6. **Selection and reporting:** best model by CV MAE; test set evaluated once; bootstrap interval on the test metrics.

## Run it

```bash
pip install -r requirements.txt
python src/train.py                                   # trains, writes results/ and models/best_model.joblib (about 40 s)
python src/predict.py data/all_in_one_dataset.csv     # example predictions with the saved model
jupyter notebook notebooks/transferiq_training.ipynb  # step-by-step walkthrough
```

```
data/       all_in_one_dataset.csv (used); legacy_data_set_final.csv (earlier preprocessed file, not used)
src/        train.py, predict.py
notebooks/  transferiq_training.ipynb (current), original_colab_notebook.ipynb (internship version)
results/    metrics.json, model_comparison.csv, plots
models/     best_model.joblib
```

## Limitations

- Only 450 players, so results have wide uncertainty.
- The player attributes look synthetically generated (for example, a player named "Erling Haaland 414" listed as a Defender at Manchester City). These results show that the pipeline works; they should not be read as real-world valuation accuracy.
- Features are a single snapshot, not a time series, so this is not a forecast of future transfer value.
- Random 80/20 split; players are independent rows, so no temporal leakage, but no external validation set either.

## Revision history

- **Dec 2025 - Jan 2026 (internship):** first version in Google Colab. XGBoost R² 0.895, Random Forest 0.911, LightGBM 0.934 (MAE EUR 2.10M, RMSE EUR 2.86M). Early stopping used the test set as its validation data, there was no cross-validation or simple baseline, and the README described steps (feature scaling, interaction features, NLP sentiment) that were not in the notebook. Kept in `notebooks/original_colab_notebook.ipynb` and `notebooks/ORIGINAL_README_for_reference.md`.
- **Nov 2026 (revision):** removed test-set leakage, added 5-fold CV, tuning, baselines, bootstrap intervals and a feature-group ablation; moved to a reproducible local script; corrected the README claims to match the evidence. The revised numbers are not directly comparable with the original ones (different protocol).
