# Phase 2 ML Implementation

## ML problem and inputs

The system is a candidate-level binary classifier. Each row is one GPS observation paired with one candidate road arc. `label=1` means the candidate arc equals the Phase 1 aligned reference arc; `label=0` means it does not. The model estimates binary candidate probabilities. HMM export normalizes those scores over each observation's candidate set.

Training data: 1,870,717 candidate rows, 80 trips, 208,417 raw GPS observations. Student-test data: 209,756 candidate rows, 10 trips, 19,160 observations.

## Schema and feature selection

Exact field types and data-quality counts are in [PHASE2_DATASET_SCHEMA.md](PHASE2_DATASET_SCHEMA.md) and `results/dataset_schema.json`. Identifier columns `trip_id`, `obs_index`, and `arc_id`, target `label`, and evaluation flag `is_gt_in_candidates` are never predictive features.

Approved Phase 1 feature columns: `snap_distance_m`, `snap_t`, `arc_length_m`, `bearing_diff_rad`, `abs_bearing_diff_rad`, `gps_speed_mps`, `gps_heading_rad`, `dt_prev_s`, `dt_next_s`, `out_degree`, `in_degree`, `is_gap_prev`, `is_gap_next`, `heading_reliable`.

All 14 approved features are numeric. Infinite values are converted to missing values; median imputation is fitted separately inside each fold. Logistic Regression and the shallow MLP also use a fold-fitted StandardScaler. No learned preprocessing is fitted on validation or student-test rows.

## Observation filtering

Initial classifier training retains observations where `is_gt_in_candidates == True`: 178,534 eligible observations and 1,742,906 candidate rows. It excludes 29,883 observations and 127,811 candidate rows. Filtering is at the observation level and preserves all candidate rows in each eligible observation.

Candidate miss rate uses all 208,417 raw training observations. There are 29,883 misses (14.338%), including 18,229 observations with an empty candidate set. This is kept separate from classifier accuracy, which is evaluated only where the true arc exists in the candidate set.

## Models and grouped validation

The five implemented models use the same approved features, target, eligible training observations, random seed, and five folds: LightGBM Classifier, Random Forest Classifier, Logistic Regression, shallow MLP Classifier, and Gaussian Naive Bayes.

The shallow MLP reached its configured 35-iteration limit on each baseline fold and emitted scikit-learn ConvergenceWarnings; the warning was retained and the measured last-iteration predictions were used.

`GroupKFold(n_splits=5)` assigns trips to folds before candidate rows are filtered. Every row from a held-out trip stays in validation; only eligible rows from training trips are fit. Automated assertions verify that training and validation trip ID sets do not overlap. The exact IDs and row counts are recorded in `results/cv_results.csv`.

Candidate Macro F1 is binary macro-averaged F1 over eligible candidate rows at a 0.5 probability threshold. Top-1 Accuracy selects the highest-scoring candidate independently for each eligible observation and checks whether that row is the aligned arc. Candidate Log Loss and Brier Score use binary candidate probabilities. Fold means and standard deviations are reported; `cv_results.csv` also has per-trip and pooled out-of-fold rows.

Candidate Miss Rate is the fraction of all raw observations with no ground-truth candidate. It includes observations with no candidate rows. Route Continuity is connected top-1 directed arc pairs divided by directly consecutive observation pairs where both observations have candidates; a pair connects when the first arc's `to_node` equals the next arc's `from_node` in that trip's graph. Arc IDs remain scoped by trip.

Training and inference times in the comparison are cumulative seconds over the five fold fits and validations. Per-fold timings are also in `results/cv_results.csv`.

## Model comparison and selection

| model | CV Macro F1 | Top-1 Accuracy | Log Loss | Candidate Miss Rate | Route Continuity | CV Training Time (s) | CV Inference Time (s) |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.8689 | 0.7957 | 0.1148 | 0.1434 | 0.3050 | 59.4504 | 6.4533 |
| random_forest | 0.8585 | 0.7993 | 0.1217 | 0.1434 | 0.3090 | 309.8877 | 4.0730 |
| logistic_regression | 0.7856 | 0.7674 | 0.1772 | 0.1434 | 0.3093 | 14.0054 | 0.2861 |
| shallow_mlp | 0.8498 | 0.7793 | 0.1291 | 0.1434 | 0.3065 | 220.2545 | 0.4274 |
| gaussian_naive_bayes | 0.7861 | 0.7576 | 0.3886 | 0.1434 | 0.3092 | 10.5383 | 0.5377 |

### Metric rankings by pooled out-of-fold predictions

The candidate-level precision and accuracy rankings use all held-out candidate predictions pooled across the same five trip-grouped folds. Macro F1, positive-class precision, and candidate binary accuracy use a 0.5 probability threshold. Top-1 accuracy selects the highest-scoring candidate per eligible observation. Higher values rank first.

| model | Candidate Macro F1 | Positive Precision | Candidate Binary Accuracy | Top-1 Accuracy | F1 Rank | Precision Rank | Candidate Accuracy Rank | Top-1 Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lightgbm | 0.8645 | 0.7822 | 0.9516 | 0.7863 | 1 | 2 | 1 | 2 |
| random_forest | 0.8544 | 0.8161 | 0.9507 | 0.7897 | 2 | 1 | 2 | 1 |
| shallow_mlp | 0.8462 | 0.7802 | 0.9470 | 0.7704 | 3 | 3 | 3 | 3 |
| gaussian_naive_bayes | 0.7851 | 0.5332 | 0.9071 | 0.7489 | 4 | 5 | 5 | 5 |
| logistic_regression | 0.7828 | 0.7127 | 0.9296 | 0.7586 | 5 | 4 | 4 | 4 |

LightGBM ranks first in pooled Macro F1 and candidate binary accuracy. Random Forest ranks first in positive precision and top-1 accuracy.

Selected baseline: **random_forest**. Selection uses a weighted mean of percentile ranks: Top-1 Accuracy (0.23), candidate Macro F1 (0.23), route continuity (0.18), candidate Log Loss (0.14), Macro F1 fold stability (0.10), five-fold training time (0.07), and inference time (0.05). Lower Log Loss, variability, and both time measures rank higher. The selection details and per-metric ranks are saved in `results/model_comparison.json`.

## Bayesian optimization

Optuna TPESampler ran 5 trials on the selected model with seed 42 (2 startup trials followed by 3 adaptive suggestions). The objective is mean five-fold GroupKFold candidate-level Log Loss on eligible validation observations; each trial also records Top-1 Accuracy. Best CV Log Loss: 0.119532; best trial: 4; Top-1 on best trial: 0.7938847432214045.

The complete trial history is in `results/optimization_trials.csv`; best parameters are in `results/best_hyperparameters.json`. `results/tuned_vs_untuned.csv` compares baseline and tuned CV performance on the same folds.

| configuration | macro_f1_mean | top1_accuracy_mean | log_loss_mean | route_continuity_pooled | training_time_s | inference_time_s |
|---|---|---|---|---|---|---|
| untuned_baseline | 0.8585 | 0.7993 | 0.1217 | 0.3090 | 309.8877 | 4.0730 |
| tuned | 0.8609 | 0.7939 | 0.1195 | 0.3068 | 795.7638 | 2.8814 |

## Probability quality and HMM interface

Probability quality is checked using held-out candidate Log Loss, Brier Score, and 10-bin expected calibration error. No post-hoc calibration model was fit. The HMM output normalization is a within-observation score normalization, not a calibration fit; it is applied independently to each trip/observation candidate group. Invalid or negative probabilities are set to zero; if all candidate scores are unusable, the group receives a uniform distribution. Empty candidate sets are represented by `candidates: []` in JSON.

Student-test output files are `results/prediction_probabilities/student_test_candidate_probabilities.parquet` and `.json`. Each Parquet row has `trip_id`, `observation_id`, `arc_id`, and `probability`. JSON contains one observation record in chronological order, with candidate probabilities that sum to one within nonempty sets. Trip ID is always part of the grouping key, so trip-specific arc IDs are not merged.

```python
import joblib, pandas as pd
model = joblib.load('artifacts/best_model.pkl')
probabilities = pd.read_parquet('results/prediction_probabilities/student_test_candidate_probabilities.parquet')
# Pass probability as the HMM emission score for its (trip_id, observation_id, arc_id).
```

## Student-test evaluation

The selected tuned estimator was fitted only on eligible training observations from the 80 training trips. Student-test labels were used once for the requested final evaluation and not for model selection or optimization.

Metrics: Top-1 Accuracy 0.8206240526838446; candidate Macro F1 0.8792235532061812; candidate Log Loss 0.09828066101076069; candidate miss rate 0.0014091858037578288; route continuity 0.26511749347258484; Brier Score 0.028835102857761048; expected calibration error 0.005487101067963079; final-model training time 215.088s; student-test inference time 1.187s.

## Saved artifacts

- `best_model`: `artifacts/best_model.pkl`
- `preprocessing_pipeline`: `artifacts/preprocessing_pipeline.pkl`
- `feature_schema`: `artifacts/feature_schema.json`
- `model_metadata`: `artifacts/model_metadata.json`
- `best_hyperparameters`: `artifacts/best_hyperparameters.json`

## Limitations

The classifier ranks individual candidates and does not enforce route continuity itself; the HMM stage should apply transition constraints. Candidate misses cannot be fixed by the classifier and remain reported separately. The supplied student-test set has only 10 trips, so its metrics are descriptive and do not replace trip-grouped cross-validation. Probability normalization creates a per-observation distribution but does not guarantee calibration under distribution shift.

## Reproduction

From the repository root, using Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Student folder.zip" -DestinationPath "data/raw"
.\.venv\Scripts\python.exe -m src.models.run_phase2 --trials 5 --seed 42
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
