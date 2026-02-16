# Adaptive Traffic Model Scaling
Model scaling analysis for adaptive traffic monitoring using YOLOv8 and a curated TrafficCAM subset.

## Overview

This project evaluates the trade-off between model size, accuracy, and inference time for traffic vehicle counting under different scene complexities.

We compare four YOLOv8 models:
- YOLOv8n (Nano)
- YOLOv8s (Small)
- YOLOv8m (Medium)
- YOLOv8l (Large)

Evaluation is performed on:
- Low Complexity traffic scenes (120 images)
- High Complexity traffic scenes (120 images)

## Evaluation Metric

Accuracy is computed using count-based matching:

accuracy = min(predicted, ground_truth) / max(predicted, ground_truth)

Average inference time per image (ms) is also recorded.

## Repository Structure

TrafficCAM/raw/Complexity/
    ├── Low_Complexity/Low_All
    └── High_Complexity/High_All

Files:
- evaluate_folder.py  → Full evaluation script
- plot_results.py     → Accuracy comparison plot
- plot_gain.py        → Relative gain over Nano
- plot_tradeoff.py    → Accuracy vs Inference time tradeoff
- count_vehicles.py   → Basic vehicle counting utility

## How to Run

Install dependencies:

pip install ultralytics matplotlib

Run evaluation:

python evaluate_folder.py

Generate plots:

python plot_results.py
python plot_gain.py
python plot_tradeoff.py

## Summary

Results demonstrate that model scaling improves accuracy more significantly in low complexity scenes than in high complexity scenes, while inference time increases substantially with model size.

This highlights the trade-off between computational cost and accuracy for adaptive traffic monitoring systems.

