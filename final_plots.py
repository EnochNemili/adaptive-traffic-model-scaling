import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import f1_score

# -------------------------------
# CONFIGURATION
# -------------------------------
# Paths provided in colleague's code and your dataset
LOW_FOLDER = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"
HIGH_FOLDER = "TrafficCAM/raw/Complexity/High_Complexity/High_All"
DAWN_FOLDER = "DAWNDataset/High_Complexity"

# Models for Comparison
MODELS_V8 = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
MODELS_V26 = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]

# GW Official Theme Colors
GW_BLUE = "#033C5A"
GW_GOLD = "#FFC72C"

# Unified Vehicle Classes for F1 standardization
#VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "vehicle"]
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "motorcycle", "vehicle", "lcv", "lmv", "auto"]

def get_gt_count(img_path, dataset_type):
    """Handles JSON (TrafficCAM) vs TXT (DAWN)"""
    if dataset_type == "trafficcam":
        json_path = img_path.replace(".jpg", ".json")
        if not os.path.exists(json_path): return 0
        with open(json_path, "r") as f:
            return len(json.load(f)["shapes"])
    else: # DAWN TXT format
        txt_path = img_path.replace(".jpg", ".txt")
        if not os.path.exists(txt_path): return 0
        with open(txt_path, "r") as f:
            return len([l for l in f if l.strip()])

def evaluate_scenario(model_list, folder_path, dataset_type):
    avg_f1s, avg_times = [], []
    images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".jpg")]

    for model_path in model_list:
        model = YOLO(model_path)
        img_f1s, img_times = [], []

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            gt_count = get_gt_count(img_path, dataset_type)
            
            # 1. Inference + Timing
            start = time.time()
            results = model(img_path, verbose=False, conf=0.25)
            img_times.append((time.time() - start) * 1000)

            # 2. Prediction Count (Vehicle-Only)
            pred_count = 0
            for box in results[0].boxes:
                label = model.names[int(box.cls)].lower()
                if any(v in label for v in VEHICLE_CLASSES):
                    pred_count += 1

            # 3. Standardized F1 Calculation
            max_len = max(gt_count, pred_count)
            if max_len == 0:
                img_f1s.append(1.0)
            else:
                y_true = [1] * gt_count + [0] * (max_len - gt_count)
                y_pred = [1] * pred_count + [0] * (max_len - pred_count)
                img_f1s.append(f1_score(y_true, y_pred, zero_division=1))

        avg_f1s.append(np.mean(img_f1s))
        avg_times.append(np.mean(img_times))
        
    return avg_f1s, avg_times

def plot_save(v8_data, v26_data, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(v26_data[1], v26_data[0], 'o-', color=GW_BLUE, label="YOLOv26 (Grant Target)", lw=2)
    plt.plot(v8_data[1], v8_data[0], 'o-', color=GW_GOLD, label="YOLOv8 (Baseline)", lw=2)
    plt.xlabel("Average Inference Time (ms)"); plt.ylabel("Vehicle-Only F1 Score")
    plt.title(title, fontweight="bold"); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{filename}.png", dpi=300); plt.savefig(f"{filename}.pdf")
    plt.close()

if __name__ == "__main__":
    scenarios = [
        (LOW_FOLDER, "trafficcam", "Low Complexity (TrafficCAM Clear)", "low_complexity"),
        (HIGH_FOLDER, "trafficcam", "High Traffic (TrafficCAM Busy)", "high_traffic"),
        (DAWN_FOLDER, "dawn", "Bad Weather (DAWN)", "bad_weather")
    ]

    for folder, d_type, title, fname in scenarios:
        print(f"Processing: {title}...")
        v8_results = evaluate_scenario(MODELS_V8, folder, d_type)
        v26_results = evaluate_scenario(MODELS_V26, folder, d_type)
        plot_save(v8_results, v26_results, title, fname)
    
    print("\nAll 3 graphs saved as PNG and PDF.")