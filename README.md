# PathSnap: Distributed Map-Matching & Route Reconstruction

A machine learning system for reconstructing realistic vehicle routes
from noisy GPS trajectories using road-network information.

## Project Overview

PathSnap matches GPS observations to road-network segments and
reconstructs coherent vehicle trajectories.

## Project Pipeline

1. Dataset exploration and preprocessing
2. Candidate road-segment generation
3. Feature engineering
4. Machine learning model training and comparison
5. Bayesian hyperparameter optimisation
6. HMM and Viterbi route reconstruction
7. FastAPI backend development
8. Evaluator-facing frontend development
9. Integration, testing and deployment

## Repository Structure

- data/ - Raw and processed datasets
- notebooks/ - Data exploration and experiments
- src/ - Core machine learning implementation
- backend/ - FastAPI application
- frontend/ - Evaluator interface
- artifacts/ - Trained models and configurations
- results/ - Experiment metrics and plots
- visualizations/ - Folium maps
- tests/ - Project tests
- report/ - Technical report

## Dataset

- 80 training trips
- 10 student-testing trips

Raw datasets are stored locally and are not committed to GitHub.

## Team

Team members and responsibilities will be added here.

## Installation

Use Python 3.12 and install the pinned-range project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The processed Phase 1 Parquet files are in `data/processed`. The raw files are
kept out of Git. To run route-continuity evaluation and the Phase 1 tests, extract
the supplied `Student folder.zip` so `data/raw/Student folder/train` and
`data/raw/Student folder/student_test` exist. From the repository root:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Student folder.zip" -DestinationPath "data/raw"
```

## Execution

Run the complete Phase 2 workflow from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.models.run_phase2 --trials 5 --seed 42
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The command compares five models with five-fold trip-grouped cross-validation,
optimizes the selected model with Optuna TPE, evaluates the ten student-test
trips, exports HMM candidate probabilities, and writes reports under `results/`
and model files under `artifacts/`. Use `--trials N` to change the optimization
budget. The default is five TPE trials; increase `--trials N` for a larger search.
Do not point the workflow at evaluator-testing data.
