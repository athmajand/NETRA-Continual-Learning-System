"""
evaluate_base_model.py
======================
NETRA MK1 - Base Model Evaluation (31 Classes)
Auto-searches all known locations for a working copy of the base model.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset, CLASSES

DEVICE = torch.device("cpu")

# ── Auto-search: tries every known location in order ──────────────────
CANDIDATE_PATHS = [
    r"D:\Threat_Detection_Workspace\03_Base_Model\base_threat_model_31_classes.pth",
    r"D:\Research Project_code\base_threat_model_31_classes.pth",
    r"D:\Research Project_code\models\base_threat_model_31_classes.pth",
    r"C:\Users\acer\Downloads\base_threat_model_31_classes.pth",
    r"D:\base_threat_model_31_classes.pth",
]
# ──────────────────────────────────────────────────────────────────────


def find_working_model():
    print("\n Searching for a working base model file...")
    for path in CANDIDATE_PATHS:
        if not os.path.exists(path):
            print("  NOT FOUND : " + path)
            continue
        size_mb = os.path.getsize(path) / 1024 / 1024
        print("  FOUND     : " + path + "  (" + str(round(size_mb, 1)) + " MB) -- trying to load...")
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            assert "analyst_head.weight" in sd, "Missing analyst_head"
            assert sd["analyst_head.weight"].shape == torch.Size([31, 128]), "Wrong shape"
            print("  LOADED OK : " + path)
            return path, sd
        except Exception as e:
            print("  CORRUPTED : " + e.__class__.__name__ + ": " + str(e)[:60])
    return None, None


def evaluate():
    print("\n" + "="*60)
    print("  NETRA MK1 - BASE MODEL EVALUATION (31 Classes)")
    print("="*60)

    model_path, sd = find_working_model()

    if sd is None:
        print("\n" + "="*60)
        print("  NO WORKING FILE FOUND")
        print("="*60)
        print("\n  All candidate paths are missing or corrupted.")
        print("  Download the clean file from this chat and save it to:")
        print(r"  D:\Research Project_code\base_threat_model_31_classes.pth")
        print("\n  Then re-run this script.")
        return

    model = FusionThreatModel(num_classes=31).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    W     = model.analyst_head.weight
    norms = W.norm(dim=1)
    print("\n  Model path : " + model_path)
    print("  Parameters : " + str(total_params))
    print("  Head norms : mean=" + str(round(norms.mean().item(), 4)) +
          "  std=" + str(round(norms.std().item(), 4)))

    print("\n Loading test dataset...")
    test_ds     = FusionDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print("  Test slices: " + str(len(test_ds)))

    per_class_correct = torch.zeros(31, dtype=torch.long)
    per_class_total   = torch.zeros(31, dtype=torch.long)
    total_correct, total_samples = 0, 0

    print("\n Running evaluation...\n")
    with torch.no_grad():
        for skel, spat, labels in tqdm(test_loader, desc="Evaluating"):
            mask = labels < 31
            if not mask.any():
                continue
            skel   = skel[mask].to(DEVICE)
            spat   = spat[mask].to(DEVICE)
            labels = labels[mask].to(DEVICE)
            _, out = model(skel, spat)
            preds  = torch.argmax(out, dim=1)
            for c in range(31):
                m = labels == c
                if m.any():
                    per_class_correct[c] += (preds[m] == c).sum().item()
                    per_class_total[c]   += m.sum().item()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0

    print("\n  Class                       Correct   Total      Acc")
    print("  " + "-"*58)
    class_accs = []
    for c in range(31):
        if per_class_total[c] > 0:
            acc_c = per_class_correct[c].item() / per_class_total[c].item() * 100
            bar   = chr(9608) * int(acc_c // 10)
            print("  " + CLASSES[c].ljust(26) +
                  str(per_class_correct[c].item()).rjust(8) +
                  str(per_class_total[c].item()).rjust(8) +
                  str(round(acc_c, 1)).rjust(8) + "%  " + bar)
            class_accs.append((CLASSES[c], acc_c))
        else:
            print("  " + CLASSES[c].ljust(26) + "  (no test samples)")

    class_accs.sort(key=lambda x: x[1], reverse=True)
    delta  = overall_acc - 89.60
    status = "CONFIRMED" if abs(delta) < 1.0 else "UNEXPECTED DELTA -- check file"

    print("\n" + "="*60)
    print("  RESULTS -- BASE MODEL (31 Classes)")
    print("="*60)
    print("  Overall Accuracy : " + str(round(overall_acc, 2)) + "%")
    print("  Expected         : 89.60%")
    print("  Delta            : " + str(round(delta, 2)) + "%")
    print("  Status           : " + status)
    print("  Total samples    : " + str(total_samples))

    print("\n  Top 5 strongest:")
    for name, acc in class_accs[:5]:
        print("    " + name.ljust(26) + str(round(acc, 1)) + "%")

    print("\n  Bottom 5 weakest:")
    for name, acc in class_accs[-5:]:
        print("    " + name.ljust(26) + str(round(acc, 1)) + "%")

    print("="*60)


if __name__ == "__main__":
    evaluate()
