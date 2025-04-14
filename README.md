# Predictive Maintenance - DevOps Pipeline with R

## Objective
Automatically detect anomalies in industrial machine data using R and GitHub Actions.

## Files
- `predictive_maintenance.csv`: Input dataset
- `anomaly_detection.R`: Anomaly detection using Z-Score
- `run_r_script.yml`: GitHub Action to automate detection and output

## How It Works
1. Push updated dataset/script to GitHub
2. GitHub Action automatically runs the R script
3. Detected anomalies are saved and downloadable from the Actions tab

## Technologies
- R (dplyr, ggplot2)
- GitHub Actions (CI/CD)
- DevOps Automation
