"""
Evaluation script for triple-riding classifier.
Computes confusion matrix, precision, recall, F1 for:
  1) 4-class: none / single / double / triple
  2) Binary:  non-triple (0,1,2) vs triple (3)

Usage:
    python evaluate.py --raw --clf_weights /path/to/new_model.ckpt   # for newly trained models
    python evaluate.py --clf_weights /path/to/tr_clf.ckpt        # for old models (uses YOLOClf +1 mapping)
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from yolo_clf import YOLOClf
from load import give_yolo_model
from lit_model import Model


# ─── Defaults ────────────────────────────────────────────────
ASSOC_WEIGHTS = "/archive/sai.teja/SAC_models/sam3_data_model/weights/best.pt"
CLF_WEIGHTS   = "/ssd_scratch/sai.teja/triple_violations/checkpoints_ogsplit_sam3/weighted/tr_clf_sam3_puf.ckpt"
VAL_DIR       = "/ssd_scratch/sai.teja/tr_dataset_ogsplit/val"
OUT_DIR       = "eval_results_sam3"
# ─────────────────────────────────────────────────────────────

RES_SUFFIX = "sam3_weighted_puf"


# Map folder names → integer labels
FOLDER_TO_LABEL = {
    "0_none":   0,
    "1_single": 1,
    "2_double": 2,
    "3_triple": 3,
}

CLASS_NAMES_4   = ["none", "single", "double", "triple"]
CLASS_NAMES_BIN = ["non-triple", "triple"]


def collect_samples(val_dir: str):
    """Return list of (image_path, label) tuples from the val directory."""
    samples = []
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(val_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠ Folder not found, skipping: {folder_path}")
            continue
        images = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg"))
            + glob.glob(os.path.join(folder_path, "*.png"))
            + glob.glob(os.path.join(folder_path, "*.jpeg"))
        )
        for img_path in images:
            samples.append((img_path, label))
    print(f"Total samples collected: {len(samples)}")
    for name, label in FOLDER_TO_LABEL.items():
        count = sum(1 for _, l in samples if l == label)
        print(f"  {name}: {count}")
    return samples


def load_raw_model(clf_weights, assoc_weights):
    """Load model directly (no YOLOClf remapping). Returns (model, transform, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = give_yolo_model(weights=assoc_weights, num_classes=4)
    model = Model(model, None, 4)
    ckpt = torch.load(clf_weights)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, transform, device


@torch.no_grad()
def run_inference_raw(model, transform, device, samples):
    """Run inference using raw model (argmax 0-3 directly = dataset labels)."""
    all_labels = []
    all_preds  = []

    for img_path, label in tqdm(samples, desc="Evaluating"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Could not read: {img_path}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(img)
        crop = transform(x)[None].to(device)
        logits = model(crop)
        pred = torch.argmax(logits[0]).item()  # raw 0-3, matches dataset labels

        all_labels.append(label)
        all_preds.append(pred)

    return np.array(all_labels), np.array(all_preds)


def run_inference(classifier, samples):
    """Run inference using YOLOClf (for old models with +1/4→0 mapping)."""
    all_labels = []
    all_preds  = []

    for img_path, label in tqdm(samples, desc="Evaluating"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Could not read: {img_path}, skipping")
            continue

        pred = classifier(img) + 1
        if pred > 3:
            # print(f"Prediction out of bounds: {pred}, setting to 3")
            pred = 3
        all_labels.append(label)
        all_preds.append(pred)

    return np.array(all_labels), np.array(all_preds)


def to_binary(labels):
    """Convert 4-class labels to binary: 0,1,2 → 0 (non-triple), 3 → 1 (triple)."""
    return (labels == 3).astype(int)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate(args):
    # ── Load model ────────────────────────────────────────────
    print("=" * 60)
    print("Loading model...")
    if args.raw:
        print("Mode: RAW (direct argmax, no +1 remapping)")
    else:
        print("Mode: YOLOClf (with +1/4→0 rider-count remapping)")
    print("=" * 60)

    # ── Collect samples ───────────────────────────────────────
    samples = collect_samples(args.val_dir)
    if len(samples) == 0:
        print("ERROR: No samples found!")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────
    print("\nRunning inference...")
    if args.raw:
        model, transform, device = load_raw_model(args.clf_weights, args.assoc_weights)
        all_labels, all_preds = run_inference_raw(model, transform, device, samples)
    else:
        classifier = YOLOClf(
            weights_path=args.clf_weights,
            rm_assoc_path=args.assoc_weights,
        )
        all_labels, all_preds = run_inference(classifier, samples)

    os.makedirs(args.out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # 4-CLASS EVALUATION
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4-CLASS EVALUATION")
    print("=" * 60)

    cm4 = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print("\nConfusion Matrix (4-class):")
    print(cm4)

    report4 = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES_4,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report (4-class):")
    print(report4)

    plot_confusion_matrix(
        cm4, CLASS_NAMES_4,
        "4-Class Confusion Matrix",
        os.path.join(args.out_dir, f"confusion_matrix_4class_{RES_SUFFIX}.png"),
    )

    # ══════════════════════════════════════════════════════════
    # BINARY EVALUATION (non-triple vs triple)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BINARY EVALUATION (non-triple vs triple)")
    print("=" * 60)

    bin_labels = to_binary(all_labels)
    bin_preds  = to_binary(all_preds)

    cm_bin = confusion_matrix(bin_labels, bin_preds, labels=[0, 1])
    print("\nConfusion Matrix (binary):")
    print(cm_bin)

    report_bin = classification_report(
        bin_labels, bin_preds,
        labels=[0, 1],
        target_names=CLASS_NAMES_BIN,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report (binary):")
    print(report_bin)

    # Extract binary metrics for triple class
    prec, rec, f1, _ = precision_recall_fscore_support(
        bin_labels, bin_preds, labels=[1], average="binary", pos_label=1,
        zero_division=0,
    )
    print(f"\n  Triple-class binary metrics:")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")

    plot_confusion_matrix(
        cm_bin, CLASS_NAMES_BIN,
        "Binary Confusion Matrix (non-triple vs triple)",
        os.path.join(args.out_dir, f"confusion_matrix_binary_{RES_SUFFIX}.png"),
    )

    # ── Save text report ──────────────────────────────────────
    report_path = os.path.join(args.out_dir, f"evaluation_report_{RES_SUFFIX}.txt")
    with open(report_path, "w") as f:
        f.write(f"Mode: {'RAW' if args.raw else 'YOLOClf'}\n")
        f.write(f"Weights: {args.clf_weights}\n\n")
        f.write("4-CLASS EVALUATION\n")
        f.write("=" * 50 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm4) + "\n\n")
        f.write(report4 + "\n\n")
        f.write("BINARY EVALUATION (non-triple vs triple)\n")
        f.write("=" * 50 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_bin) + "\n\n")
        f.write(report_bin + "\n")
        f.write(f"\nTriple-class binary metrics:\n")
        f.write(f"  Precision : {prec:.4f}\n")
        f.write(f"  Recall    : {rec:.4f}\n")
        f.write(f"  F1 Score  : {f1:.4f}\n")
    print(f"\n  Full report saved: {report_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate triple-riding classifier")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR,
                        help="Path to val directory with 0_none/1_single/2_double/3_triple subfolders")
    parser.add_argument("--assoc_weights", type=str, default=ASSOC_WEIGHTS,
                        help="Path to rm_assoc model weights")
    parser.add_argument("--clf_weights", type=str, default=CLF_WEIGHTS,
                        help="Path to classifier checkpoint")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw argmax (for newly trained models). "
                             "Without this flag, uses YOLOClf +1/4→0 mapping (for old models).")
    args = parser.parse_args()
    evaluate(args)
