import os
import xml.etree.ElementTree as ET
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------
LOW_COMPLEXITY_DIR = "DAWNDataset/Low_Complexity"
HIGH_COMPLEXITY_DIR = "DAWNDataset/High_Complexity"
LIMIT = 67  # Ensuring balanced 1:1 comparison

MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "motorcycle"]

# -------------------------------
# GROUND TRUTH PARSERS
# -------------------------------

def get_txt_ground_truth(txt_path):
    """Counts lines in a YOLO .txt file."""
    if not os.path.exists(txt_path): return 0
    with open(txt_path, "r") as f:
        return sum(1 for line in f if line.strip())

def get_xml_ground_truth(xml_path, filename):
    """Counts <target> tags for a specific frame in UA-DETRAC XML."""
    if not os.path.exists(xml_path): return 0
    try:
        # Assuming filename format img00001.jpg -> extract '1'
        frame_num = str(int(''.join(filter(str.isdigit, filename))))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frame = root.find(f".//frame[@num='{frame_num}']")
        if frame is not None:
            target_list = frame.find('target_list')
            return len(target_list.findall('target')) if target_list is not None else 0
    except Exception as e:
        print(f"Error parsing XML for {filename}: {e}")
    return 0

# -------------------------------
# EVALUATION ENGINE
# -------------------------------

def evaluate_set(model, folder_path, is_xml=False):
    total_acc = 0
    count = 0
    
    # Filter for jpg files and limit to 67
    images = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".jpg")][:LIMIT]

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        
        # Get Ground Truth
        if is_xml:
            # Assumes one .xml file exists in the folder for the sequence
            xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
            xml_path = os.path.join(folder_path, xml_files[0]) if xml_files else ""
            gt = get_xml_ground_truth(xml_path, img_name)
        else:
            txt_path = img_path.replace(".jpg", ".txt")
            gt = get_txt_ground_truth(txt_path)

        # Run Inference
        results = model(img_path, verbose=False, conf=0.5) # conf=0.5 to reduce noise
        
        pred = 0
        for cls_id in results[0].boxes.cls:
            if model.names[int(cls_id)] in VEHICLE_CLASSES:
                pred += 1

        # Accuracy Ratio
        if max(pred, gt) > 0:
            total_acc += min(pred, gt) / max(pred, gt)
        else:
            total_acc += 1.0
        
        count += 1

    return total_acc / count if count > 0 else 0

# -------------------------------
# MAIN EXECUTION
# -------------------------------

low_results = []
high_results = []
model_labels = ["Nano", "Small", "Medium", "Large"]

for m_name in MODELS:
    print(f"Evaluating {m_name}...")
    model = YOLO(m_name)
    
    low_acc = evaluate_set(model, LOW_COMPLEXITY_DIR, is_xml=True)
    high_acc = evaluate_set(model, HIGH_COMPLEXITY_DIR, is_xml=False)
    
    low_results.append(low_acc)
    high_results.append(high_acc)

# -------------------------------
# PLOTTING
# -------------------------------

plt.figure(figsize=(8, 5))
plt.plot(model_labels, low_results, marker='o', label="Low Complexity (UA-DETRAC)", color='blue')
plt.plot(model_labels, high_results, marker='o', label="High Complexity (DAWN)", color='red')

plt.title(f"Model Size vs Accuracy (n={LIMIT} balanced)")
plt.xlabel("Model Size")
plt.ylabel("Accuracy Ratio")
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig("new_experiment_results.png")
plt.show()

print("\nDone! Graph saved as 'new_experiment_results.png'")