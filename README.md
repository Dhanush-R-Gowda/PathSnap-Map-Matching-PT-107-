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

Instructions will be added after dependencies are finalised.

## Execution

Instructions will be added as components are developed.
