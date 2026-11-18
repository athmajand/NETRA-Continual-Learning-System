"""
fast_accuracy_evaluator.py
========================
Evaluates both Watchdog and Analyst heads instantly.
Filters out empty/removed classes for a clean thesis report.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset

# ================= CONFIGURATION =================
MODEL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\netra_mk2_headtune_32classes.pth"
DEVICE = torch.device("cpu")

# The original 32 classes the model was mathematically trained on
FULL_32_CLASSES = [
    'Normal', 'Fighting', 'Burglary', 'Arson', 'Shooting',
    'Abuse', 'Arrest', 'Assault', 'Explosion', 'Fall',
    'FenceClimbing', 'Kidnapping', 'Peeping', 'RoadAccidents', 'Robbery',
    'Shoplifting', 'slap', 'Snatching', 'stab', 'Stealing',
    'SuspiciousBehavior', 'Vandalism', 'Vehicle_Intrusion', 'medical_emergency',
    'Weaponized', 'weapon_brandishing', 'intoxicated_behaviour', 'Crowd_Anomaly',
    'atm_theft', 'Animal_Attack', 'Stone_Pelting', 'Package_Theft' 
]

# Create filters to completely hide 'slap' and 'stab' from the final printed report
VALID_INDICES = [i for i, name in enumerate(FULL_32_CLASSES) if name not in ['slap', 'stab']]
VALID_CLASSES = [name for name in FULL_32_CLASSES if name not in ['slap', 'stab']]
# =================================================

def evaluate_fast():
    print("\n" + "="*70)
    print("  NETRA MK-I — FAST DEPLOYMENT METRICS EVALUATION")
    print("="*70)

    # 1. Load model (Must remain 32 to prevent size mismatch crash)
    print("\nLoading NETRA Mk-I Model...")
    model = FusionThreatModel(num_classes=32).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure MODEL_PATH is correct.")
        return
    model.eval()

    # 2. Load test set using the fast FusionDataset
    print("\nLoading pre-extracted Test Data (Super Fast)...")
    try:
        test_full = FusionDataset(split="test")
        test_loader = DataLoader(test_full, batch_size=32, shuffle=False, num_workers=0)
        print(f"Dataset loaded! Total test slices: {len(test_full)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Tracking arrays
    analyst_preds = []
    analyst_labels = []
    watchdog_preds = []
    watchdog_labels = []

    print("\nRunning inference...")
    with torch.no_grad():
        for skel, spat, labels in tqdm(test_loader, desc="Evaluating"):
            skel, spat, labels = skel.to(DEVICE), spat.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            out_binary, out_fine = model(skel, spat)
            
            # Analyst Head (32-Class Granular)
            _, a_preds = torch.max(out_fine, dim=1)
            analyst_preds.extend(a_preds.cpu().numpy())
            analyst_labels.extend(labels.cpu().numpy())

            # Watchdog Head (Binary: Threat vs Normal)
            _, w_preds = torch.max(out_binary, dim=1)
            watchdog_preds.extend(w_preds.cpu().numpy())
            
            # Create True Binary Labels: 0 if Normal (index 0), 1 if Threat
            w_labels = (labels != 0).long() 
            watchdog_labels.extend(w_labels.cpu().numpy())

    # ================= FINAL REPORT GENERATION =================
    print("\n\n" + "="*70)
    print("FINAL NETRA Mk-I DEPLOYMENT METRICS")
    print("="*70)
    
    # 1. Watchdog Binary Accuracy
    watch_acc = accuracy_score(watchdog_labels, watchdog_preds) * 100
    print(f"🛡️ WATCHDOG ENGINE (Threat vs. Normal) ACCURACY : {watch_acc:.2f}%\n")
    
    # 2. Analyst Class-Wise Accuracy (Filtered for 30 classes)
    print("🎯 ANALYST ENGINE (Granular Identification) REPORT :")
    report = classification_report(
        analyst_labels, 
        analyst_preds, 
        labels=VALID_INDICES, 
        target_names=VALID_CLASSES, 
        zero_division=0
    )
    print(report)
    print("="*70)

if __name__ == "__main__":
    evaluate_fast()