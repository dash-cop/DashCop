"""
CAS Association Score Evaluator
================================
Evaluates how well the trained CAS model associates riders with their motorcycles.

For each image in the validation set:
  1. Runs model inference to get regular masks + cross-association masks
  2. Computes an IoU-based association matrix between rider cross-masks and motor masks
  3. Uses the ground-truth assoc_ids to check if the predicted best-match is correct

Metrics reported:
  - Association Accuracy: fraction of riders correctly matched to their GT motorcycle
  - Mean Association IoU: average IoU of the correct rider→motor pairs
  - Per-image association matrix printed (optional with --verbose)

Usage:
  python eval_association.py --model <path_to_model.pt> --data <path_to_val_folder> [--device 0] [--conf 0.25] [--verbose]

The validation folder should follow the CAS format:
  val/
    image1.jpg
    image1.txt   # cls assoc_id x1 y1 x2 y2 ...
    image2.jpg
    image2.txt
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch

# Add the local ultralytics to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultralytics"))
from ultralytics import YOLO


def parse_gt_annotations(txt_path, img_h, img_w):
    """
    Parse a CAS-format annotation file.

    Returns:
        list of dicts with keys: cls, assoc_id, mask_polygon (pixel coords), bbox (x1,y1,x2,y2)
    """
    annotations = []
    if not os.path.exists(txt_path):
        return annotations

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id = int(parts[0])
            assoc_id = int(parts[1])
            coords = list(map(float, parts[2:]))

            # Convert normalized polygon coords to pixel coords
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]

            bbox = [min(xs), min(ys), max(xs), max(ys)]
            polygon = np.array(list(zip(xs, ys)), dtype=np.float32)

            annotations.append({
                "cls": cls_id,
                "assoc_id": assoc_id,
                "polygon": polygon,
                "bbox": bbox,
            })

    return annotations


def bbox_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes, pred_classes, gt_annotations, iou_thresh=0.5):
    """
    Match predicted detections to ground truth annotations using IoU.
    
    Returns:
        matches: list of (pred_idx, gt_idx) pairs
    """
    matches = []
    used_gt = set()
    used_pred = set()

    # Build IoU matrix
    n_pred = len(pred_boxes)
    n_gt = len(gt_annotations)
    if n_pred == 0 or n_gt == 0:
        return matches

    iou_matrix = np.zeros((n_pred, n_gt))
    for pi in range(n_pred):
        for gi in range(n_gt):
            # Only match same class
            if pred_classes[pi] != gt_annotations[gi]["cls"]:
                continue
            iou_matrix[pi, gi] = bbox_iou(pred_boxes[pi], gt_annotations[gi]["bbox"])

    # Greedy matching by highest IoU
    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break
        pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches.append((int(pi), int(gi)))
        iou_matrix[pi, :] = 0
        iou_matrix[:, gi] = 0

    return matches


def compute_association_iou(masks_a, masks_b):
    """
    Compute pairwise mask IoU between two sets of binary masks.

    Args:
        masks_a: tensor of shape (N, H, W)
        masks_b: tensor of shape (M, H, W)

    Returns:
        iou_matrix: tensor of shape (N, M)
    """
    N = masks_a.shape[0]
    M = masks_b.shape[0]

    masks_a_flat = masks_a.reshape(N, -1).float()  # (N, HW)
    masks_b_flat = masks_b.reshape(M, -1).float()  # (M, HW)

    # (N, 1, HW) * (1, M, HW) => (N, M, HW)
    intersection = (masks_a_flat[:, None, :] * masks_b_flat[None, :, :]).sum(-1)  # (N, M)
    union_map = masks_a_flat[:, None, :] + masks_b_flat[None, :, :]               # (N, M, HW)
    union_map = (union_map > 0).float().sum(-1)                                    # (N, M)

    iou = intersection / (union_map + 1e-8)  # (N, M)
    return iou


def evaluate_association(model_path, data_dir, device="0", conf=0.25, iou_thresh=0.5, verbose=False):
    """
    Main evaluation function.

    For each image:
      1. Run model inference
      2. Match predicted detections to GT (by bbox IoU and class)
      3. For matched riders: compute cross-mask IoU with all matched motor regular masks
      4. Check if the argmax of the cross-mask IoU corresponds to the GT assoc_id

    Returns a dict with evaluation metrics.
    """
    model = YOLO(model_path)

    # Gather images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    image_files.sort()

    if len(image_files) == 0:
        print(f"No images found in {data_dir}")
        return

    print(f"Found {len(image_files)} images in {data_dir}")
    print(f"Model: {model_path}")
    print(f"Device: {device}, Confidence: {conf}, IoU thresh: {iou_thresh}")
    print("=" * 70)

    # Accumulators
    total_riders_with_assoc = 0          # riders that have a GT association
    correct_associations = 0             # riders matched to correct motor
    all_correct_ious = []                # IoU of correctly matched pairs
    all_predicted_ious = []              # IoU of predicted best-match pairs
    total_unmatched_riders = 0           # riders in GT but not detected
    total_unmatched_motors = 0           # motors in GT but not detected
    images_with_associations = 0
    total_motors_with_assoc = 0
    correct_motor_associations = 0

    for img_path in tqdm(image_files,desc="processing images"):
        img_name = Path(img_path).stem
        txt_path = os.path.join(data_dir, img_name + ".txt")

        # Load image for dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue
        img_h, img_w = img.shape[:2]

        # Parse GT
        gt_annots = parse_gt_annotations(txt_path, img_h, img_w)
        if len(gt_annots) == 0:
            continue

        # Run inference
        results = model.predict(img_path, conf=conf, device=device, verbose=False)
        if len(results) == 0:
            continue

        r = results[0]

        # Check we have predictions with masks
        if r.boxes is None or len(r.boxes) == 0:
            # Count unmatched GT
            for ann in gt_annots:
                if ann["cls"] == 0:
                    total_unmatched_riders += 1
                elif ann["cls"] == 1:
                    total_unmatched_motors += 1
            continue

        if r.masks is None or r.masks_cross is None:
            continue

        pred_boxes_data = r.boxes.data.cpu()
        pred_boxes = pred_boxes_data[:, :4].numpy()        # (N, 4) xyxy
        pred_confs = pred_boxes_data[:, 4].numpy()         # (N,)
        pred_classes = pred_boxes_data[:, 5].numpy().astype(int)  # (N,)
        pred_masks = r.masks.data.cpu()                    # (N, H, W)
        pred_masks_cross = r.masks_cross.data.cpu()        # (N, H, W)

        # Identify riders and motors in predictions
        rider_pred_idxs = np.where(pred_classes == 0)[0]
        motor_pred_idxs = np.where(pred_classes == 1)[0]

        if len(rider_pred_idxs) == 0 or len(motor_pred_idxs) == 0:
            continue

        # Match predictions to GT
        matches = match_predictions_to_gt(pred_boxes, pred_classes, gt_annots, iou_thresh=iou_thresh)

        # Build mapping: pred_idx -> gt_idx
        pred_to_gt = {m[0]: m[1] for m in matches}
        gt_to_pred = {m[1]: m[0] for m in matches}

        # Build GT association groups: assoc_id -> {riders: [gt_idx], motors: [gt_idx]}
        assoc_groups = {}
        for gi, ann in enumerate(gt_annots):
            aid = ann["assoc_id"]
            if aid not in assoc_groups:
                assoc_groups[aid] = {"riders": [], "motors": []}
            if ann["cls"] == 0:
                assoc_groups[aid]["riders"].append(gi)
            elif ann["cls"] == 1:
                assoc_groups[aid]["motors"].append(gi)

        # Get cross-mask IoU matrix: rider_cross_masks vs motor_regular_masks
        rider_cross_masks = pred_masks_cross[rider_pred_idxs]  # (R, H, W) - what motorcycle each rider predicts
        motor_regular_masks = pred_masks[motor_pred_idxs]      # (M, H, W) - actual motorcycle masks

        # Compute association IoU matrix (R x M)
        assoc_iou = compute_association_iou(rider_cross_masks, motor_regular_masks)  # (R, M)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Image: {img_name}")
            print(f"  Riders detected: {len(rider_pred_idxs)}, Motors detected: {len(motor_pred_idxs)}")
            print(f"  Association IoU matrix (riders × motors):")
            print(f"  {assoc_iou.numpy()}")

        has_assoc = False

        # For each matched rider, check if its predicted best motor matches GT
        for local_r_idx, pred_r_idx in enumerate(rider_pred_idxs):
            if pred_r_idx not in pred_to_gt:
                continue  # unmatched prediction
            
            gt_r_idx = pred_to_gt[pred_r_idx]
            gt_rider_ann = gt_annots[gt_r_idx]
            gt_assoc_id = gt_rider_ann["assoc_id"]

            # Find GT motor(s) for this rider's assoc_id
            if gt_assoc_id not in assoc_groups:
                continue
            gt_motor_idxs = assoc_groups[gt_assoc_id]["motors"]
            if len(gt_motor_idxs) == 0:
                continue

            total_riders_with_assoc += 1
            has_assoc = True

            # Which motor did the model predict as best match?
            iou_row = assoc_iou[local_r_idx]  # (M,)
            predicted_local_m_idx = torch.argmax(iou_row).item()
            predicted_pred_m_idx = motor_pred_idxs[predicted_local_m_idx]
            predicted_iou = iou_row[predicted_local_m_idx].item()
            all_predicted_ious.append(predicted_iou)

            # Check if the predicted motor matches any GT motor with same assoc_id
            is_correct = False
            if predicted_pred_m_idx in pred_to_gt:
                predicted_gt_m_idx = pred_to_gt[predicted_pred_m_idx]
                if predicted_gt_m_idx in gt_motor_idxs:
                    is_correct = True

            if is_correct:
                correct_associations += 1
                all_correct_ious.append(predicted_iou)

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"  Rider pred#{pred_r_idx} (GT assoc={gt_assoc_id}) → Motor pred#{predicted_pred_m_idx} "
                      f"(IoU={predicted_iou:.3f}) [{status}]")

        # Also evaluate motor → rider association (reverse direction)
        motor_cross_masks_for_motors = pred_masks_cross[motor_pred_idxs]  # Cross masks for motors predict rider masks
        rider_regular_masks = pred_masks[rider_pred_idxs]

        # Motor cross-mask IoU with rider regular masks: (M, R)
        motor_assoc_iou = compute_association_iou(motor_cross_masks_for_motors, rider_regular_masks)

        for local_m_idx, pred_m_idx in enumerate(motor_pred_idxs):
            if pred_m_idx not in pred_to_gt:
                continue
            gt_m_idx = pred_to_gt[pred_m_idx]
            gt_motor_ann = gt_annots[gt_m_idx]
            gt_assoc_id = gt_motor_ann["assoc_id"]

            if gt_assoc_id not in assoc_groups:
                continue
            gt_rider_idxs = assoc_groups[gt_assoc_id]["riders"]
            if len(gt_rider_idxs) == 0:
                continue

            total_motors_with_assoc += 1

            # Which rider did the model predict as best match?
            iou_row = motor_assoc_iou[local_m_idx]
            predicted_local_r_idx = torch.argmax(iou_row).item()
            predicted_pred_r_idx = rider_pred_idxs[predicted_local_r_idx]

            is_correct = False
            if predicted_pred_r_idx in pred_to_gt:
                predicted_gt_r_idx = pred_to_gt[predicted_pred_r_idx]
                if predicted_gt_r_idx in gt_rider_idxs:
                    is_correct = True

            if is_correct:
                correct_motor_associations += 1

        if has_assoc:
            images_with_associations += 1

    # Print summary
    print("\n" + "=" * 70)
    print("ASSOCIATION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Images evaluated: {len(image_files)}")
    print(f"Images with associations: {images_with_associations}")
    print()

    print("--- Rider → Motor Association ---")
    if total_riders_with_assoc > 0:
        acc = correct_associations / total_riders_with_assoc
        print(f"  Total riders with GT association: {total_riders_with_assoc}")
        print(f"  Correctly associated:             {correct_associations}")
        print(f"  Association Accuracy:             {acc:.4f} ({acc*100:.1f}%)")
        if len(all_predicted_ious) > 0:
            print(f"  Mean predicted IoU (all):         {np.mean(all_predicted_ious):.4f}")
        if len(all_correct_ious) > 0:
            print(f"  Mean predicted IoU (correct):     {np.mean(all_correct_ious):.4f}")
    else:
        print("  No rider associations found to evaluate.")

    print()
    print("--- Motor → Rider Association ---")
    if total_motors_with_assoc > 0:
        acc_m = correct_motor_associations / total_motors_with_assoc
        print(f"  Total motors with GT association: {total_motors_with_assoc}")
        print(f"  Correctly associated:             {correct_motor_associations}")
        print(f"  Association Accuracy:             {acc_m:.4f} ({acc_m*100:.1f}%)")
    else:
        print("  No motor associations found to evaluate.")

    print("=" * 70)

    return {
        "rider_to_motor_accuracy": correct_associations / max(total_riders_with_assoc, 1),
        "motor_to_rider_accuracy": correct_motor_associations / max(total_motors_with_assoc, 1),
        "total_riders_evaluated": total_riders_with_assoc,
        "total_motors_evaluated": total_motors_with_assoc,
        "mean_predicted_iou": np.mean(all_predicted_ious) if all_predicted_ious else 0,
        "mean_correct_iou": np.mean(all_correct_ious) if all_correct_ious else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAS model association quality")
    parser.add_argument("--model", type=str,
                        default="/ssd_scratch/cvit/saiteja/DashCop_ckpts/rm_assoc_model/model_ft.pt",
                        help="Path to trained CAS model (.pt)")
    parser.add_argument("--data", type=str,
                        default="/ssd_scratch/cvit/saiteja/SAC_dataset/val",
                        help="Path to validation folder with images and CAS annotation .txt files")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to run inference on (default: '0')")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for matching predictions to GT (default: 0.5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-image association details")

    args = parser.parse_args()

    metrics = evaluate_association(
        model_path=args.model,
        data_dir=args.data,
        device=args.device,
        conf=args.conf,
        iou_thresh=args.iou_thresh,
        verbose=args.verbose,
    )
