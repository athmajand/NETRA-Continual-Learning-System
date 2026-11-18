"""
audit_all_checkpoints.py
========================
NETRA MK1 - Evaluate ALL checkpoint files in one run.
Run: python audit_all_checkpoints.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix: multiple OpenMP runtime conflict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_model import FusionThreatModel
from fusion_dataloader import FusionDataset, CLASSES

# ══════════════════════════════════════════════════════════════════════
CHECKPOINT_DIR = r"D:\Threat_Detection_Workspace\03_Base_Model"
BASE_PATH      = os.path.join(CHECKPOINT_DIR, "base_threat_model_31_classes.pth")
NEW_CLASS_IDX  = 31
NUM_OLD        = 31
DEVICE         = torch.device("cpu")

FILES_TO_AUDIT = [
    "base_threat_model_31_classes.pth"
]
# ══════════════════════════════════════════════════════════════════════


def try_load(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model = FusionThreatModel(num_classes=32).to(DEVICE)
        model.load_state_dict(sd, strict=True)
        return model, sd, None
    except Exception:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            model = FusionThreatModel(num_classes=31).to(DEVICE)
            model.load_state_dict(sd, strict=True)
            return model, sd, "31-class model"
        except Exception as e2:
            return None, None, str(e2)[:80]


def weight_audit(sd):
    if sd is None or "analyst_head.weight" not in sd:
        return {}
    W = sd["analyst_head.weight"]
    b = sd["analyst_head.bias"]
    n = W.shape[0]
    out = {"n_classes": n}
    if n == 32:
        old_norms = W[:31].norm(dim=1)
        new_norm  = W[31].norm()
        out["old_norm"] = old_norms.mean().item()
        out["new_norm"] = new_norm.item()
        out["ratio"]    = (old_norms.mean() / (new_norm + 1e-8)).item()
        out["new_bias"] = b[31].item()
    return out


def check_iron_shield(sd, base_sd):
    if sd is None:
        return "N/A"
    changed = [k for k, v in base_sd.items()
               if "analyst_head" not in k and k in sd and not torch.equal(v, sd[k])]
    return "INTACT" if not changed else str(len(changed)) + " layers drifted"


def evaluate_model(model, test_loader):
    model.eval()
    c_old, t_old, c_new, t_new = 0, 0, 0, 0
    n_classes = model.analyst_head.out_features
    with torch.no_grad():
        for skel, spat, labels in tqdm(test_loader, desc="  Evaluating", leave=False):
            skel, spat = skel.to(DEVICE), spat.to(DEVICE)
            if n_classes == 31:
                mask = labels < 31
                if not mask.any():
                    continue
                skel, spat, labels = skel[mask], spat[mask], labels[mask]
            labels = labels.to(DEVICE)
            _, out = model(skel, spat)
            preds  = torch.argmax(out, dim=1)
            if n_classes == 32:
                m_old = labels < NUM_OLD
                m_new = labels == NEW_CLASS_IDX
                if m_old.any():
                    c_old += (preds[m_old] == labels[m_old]).sum().item()
                    t_old += m_old.sum().item()
                if m_new.any():
                    c_new += (preds[m_new] == labels[m_new]).sum().item()
                    t_new += m_new.sum().item()
            else:
                c_old += (preds == labels).sum().item()
                t_old += labels.size(0)
    core = c_old / t_old * 100 if t_old > 0 else 0.0
    new  = c_new / t_new * 100 if t_new > 0 else 0.0
    return core, new, int(t_new)


def main():
    print("\n" + "="*65)
    print("  NETRA MK1 — FULL CHECKPOINT AUDIT")
    print("="*65)

    base_sd = torch.load(BASE_PATH, map_location="cpu", weights_only=False)

    print("\nLoading test dataset (shared across all files)...")
    test_full   = FusionDataset(split="test")
    test_loader = DataLoader(test_full, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test slices: {len(test_full)}\n")

    results = []

    for fname in FILES_TO_AUDIT:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        print("\n" + "-"*65)
        print("FILE: " + fname)
        print("-"*65)

        if not os.path.exists(fpath):
            print("  NOT FOUND — skipping")
            results.append({"file": fname, "status": "NOT FOUND",
                            "core": -1, "new": -1})
            continue

        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print("  Size: " + str(round(size_mb, 1)) + " MB")

        model, sd, err = try_load(fpath)

        if model is None:
            print("  LOAD FAILED: " + str(err))
            results.append({"file": fname, "status": "LOAD FAILED",
                            "core": -1, "new": -1})
            continue

        wa     = weight_audit(sd)
        shield = check_iron_shield(sd, base_sd)
        n_cls  = wa.get("n_classes", "?")
        print("  Classes    : " + str(n_cls))

        if wa.get("n_classes") == 32:
            print("  Old norm   : " + str(round(wa["old_norm"], 4)) +
                  "  New norm: " + str(round(wa["new_norm"], 4)) +
                  "  Ratio: " + str(round(wa["ratio"], 2)) + "x")

        print("  Iron Shield: " + shield)
        print("  Evaluating (this takes ~40 min)...")

        core, new_acc, n_test = evaluate_model(model, test_loader)
        forgetting = 89.60 - core
        pkg_n = int(round(new_acc / 100 * n_test)) if n_test > 0 else 0

        print("  Core Memory   : " + str(round(core, 2)) + "%")
        if n_cls == 32:
            print("  Package_Theft : " + str(round(new_acc, 2)) +
                  "% (" + str(pkg_n) + "/" + str(n_test) + " clips)")
        print("  Forgetting    : " + str(round(forgetting, 2)) + "%")

        if core >= 85:
            verdict = "BEST CANDIDATE"
        elif core >= 80:
            verdict = "CANDIDATE"
        else:
            verdict = "DELETE"

        results.append({
            "file"      : fname,
            "status"    : "OK",
            "classes"   : n_cls,
            "core"      : core,
            "new"       : new_acc,
            "n_test"    : n_test,
            "shield"    : shield,
            "forgetting": forgetting,
            "old_norm"  : wa.get("old_norm", 0),
            "new_norm"  : wa.get("new_norm", 0),
            "ratio"     : wa.get("ratio", 0),
            "verdict"   : verdict,
        })

    # ── FINAL TABLE ───────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  AUDIT COMPLETE — RANKED RESULTS")
    print("="*65)
    print("  {:<44} {:>7} {:>7} {:>9}  {}".format(
          "File", "Core", "New", "Forget", "Verdict"))
    print("  " + "-"*62)

    valid = [r for r in results if r["status"] == "OK"]
    valid.sort(key=lambda x: x["core"], reverse=True)

    for r in valid:
        print("  {:<44} {:>6.2f}% {:>6.2f}% {:>+8.2f}%  {}".format(
              r["file"], r["core"], r["new"], r["forgetting"], r["verdict"]))

    for r in results:
        if r["status"] != "OK":
            print("  {:<44} {:>7} {:>7} {:>9}  {}".format(
                  r["file"], "-", "-", "-", r["status"]))

    if valid:
        best = valid[0]
        print("\n  KEEP THIS FILE:")
        print("  " + best["file"])
        print("  Core=" + str(round(best["core"], 2)) +
              "%  PkgTheft=" + str(round(best["new"], 2)) +
              "%  Forgetting=" + str(round(best["forgetting"], 2)) + "%")

        print("\n  SAFE TO DELETE:")
        for r in results:
            if r["core"] < 80 or r["status"] != "OK":
                print("  " + r["file"] + " (" + r["status"] + ")")

    # Save log
    log_path = os.path.join(CHECKPOINT_DIR, "checkpoint_audit_log.txt")
    with open(log_path, "w") as f:
        f.write("NETRA MK1 CHECKPOINT AUDIT\n")
        f.write("="*65 + "\n\n")
        for r in results:
            f.write("File    : " + r["file"] + "\n")
            f.write("Status  : " + r["status"] + "\n")
            if r["status"] == "OK":
                f.write("Core    : " + str(round(r["core"], 2)) + "%\n")
                f.write("PkgTheft: " + str(round(r["new"], 2)) + "%\n")
                f.write("Forget  : " + str(round(r["forgetting"], 2)) + "%\n")
                f.write("Shield  : " + r["shield"] + "\n")
                f.write("Verdict : " + r["verdict"] + "\n")
            f.write("\n")
    print("\n  Audit log saved: " + log_path)
    print("="*65)


if __name__ == "__main__":
    main()
