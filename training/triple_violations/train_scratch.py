"""
Plain PyTorch training script for triple-riding classifier (no Lightning).

Usage:
    python train_scratch.py                           # uses defaults from cfg.yaml
    python train_scratch.py --epochs 100 --lr 0.001
    python train_scratch.py --resume /path/to/ckpt    # resume training

    python train_scratch.py 2>&1 | tee train_scratch.log
"""

import os
import sys
import argparse
import yaml
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from load import give_yolo_model
from lit_model import Model
from dataset import TripleRidingDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEFAULT_CFG = "cfg.yaml"


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """Build model with frozen BN handling matching the working diagnostic."""
    sac_weights = cfg.get("sac_weights", "/archive/sai.teja/rm_assoc_model/model_ft.pt")
    num_classes = cfg.get("nc", 4)
    lr = cfg.get("lr", 0.001)

    print(f"Building model from backbone: {sac_weights}")
    yolo_model = give_yolo_model(weights=sac_weights, num_classes=num_classes)

    # Handle freeze/unfreeze
    unfreeze_all = cfg.get("unfreeze_all", False)
    freeze_all = cfg.get("freeze_all_except_classifier", False)

    if unfreeze_all:
        print("Unfreezing ALL layers for full fine-tuning")
        for param in yolo_model.parameters():
            param.requires_grad = True
    elif freeze_all:
        print("Freezing all layers except classifier")
        for name, param in yolo_model.model.named_parameters():
            param.requires_grad = not ("classifier" not in name)

    # Print trainable param summary
    trainable = [p for p in yolo_model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    print(f"Total parameters:     {sum(p.numel() for p in yolo_model.parameters()):,}")

    return yolo_model, lr


def set_bn_eval(model):
    """Set frozen BatchNorm layers to eval mode (use pretrained running stats).
    Trainable BN (cv4, layer 15) stays in train mode (batch stats).
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            all_frozen = all(not p.requires_grad for p in module.parameters())
            if all_frozen:
                module.eval()


def build_dataloaders(cfg):
    data_root = cfg.get("path", "/ssd_scratch/sai.teja/bal_tr_dataset")
    batch_size = cfg.get("batch_size", 32)
    imgsz = cfg.get("imgsz", 224)
    num_workers = cfg.get("num_workers", 4)
    use_weighted_sampling = cfg.get("weighted_sampling", False)

    train_dataset = TripleRidingDataset(data_root, split="train", imgsz=imgsz)
    val_dataset = TripleRidingDataset(data_root, split="val", imgsz=imgsz)

    sampler = None
    shuffle = True
    if use_weighted_sampling:
        labels = np.array(train_dataset.labels)
        class_counts = np.bincount(labels, minlength=cfg.get("nc", 4))
        class_weights = 1.0 / class_counts.astype(np.float64)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
        print("Weighted sampling enabled:")
        for i, (cnt, w) in enumerate(zip(class_counts, class_weights)):
            print(f"  Class {i}: count={cnt}, weight={w:.6f}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    return train_loader, val_loader


@torch.no_grad()
def validate(model, val_loader, criterion, device, num_classes):
    """Run validation and return metrics."""
    model.eval()
    val_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        val_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c] += mask.sum().item()

    val_loss /= sum(class_total)
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    val_acc = total_correct / total_samples if total_samples > 0 else 0

    class_accs = {}
    for c in range(num_classes):
        class_accs[c] = class_correct[c] / class_total[c] if class_total[c] > 0 else 0

    return val_loss, val_acc, class_accs


def train(args):
    cfg = load_config(args.cfg)

    # CLI overrides
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.lr is not None: cfg["lr"] = args.lr
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.device is not None: cfg["device"] = args.device

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 10)
    ckpt_dir = cfg.get("checkpoint_dir", "/ssd_scratch/sai.teja/triple_violations/checkpoints")
    ckpt_name = cfg.get("checkpoint_name", "tr_clf.ckpt")
    device_id = cfg.get("device", 0)
    num_classes = cfg.get("nc", 4)

    if str(device_id) != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and str(device_id) != "cpu" else "cpu")

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ── Build model ───────────────────────────────────────────
    yolo_model, lr = build_model(cfg)
    # Wrap in LightningModule (for checkpoint compatibility)
    trainable = [p for p in yolo_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model = Model(yolo_model, optimizer, num_classes, lr=lr)
    model.to(device)

    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        # Re-create optimizer with loaded params
        trainable = [p for p in model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    # ── Build dataloaders ─────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Wandb ─────────────────────────────────────────────────
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or ckpt_name.replace(".ckpt", ""),
            config=cfg,
        )
        # Use "epoch" as x-axis for all metrics so charts align
        # with the Lightning run (which also logs per-epoch metrics)
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # ── Training loop ─────────────────────────────────────────
    best_val_acc3 = 0.0
    best_epoch = -1
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs on {device}...")
    print(f"Early stopping patience: {patience}\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────
        model.train()
        set_bn_eval(model.model)  # Frozen BN → eval, trainable BN → train

        train_loss = 0.0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            for c in range(num_classes):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c] += mask.sum().item()

        scheduler.step()

        # Compute train metrics
        total_train = sum(class_total)
        train_loss_avg = train_loss / total_train if total_train > 0 else 0
        train_acc = sum(class_correct) / total_train if total_train > 0 else 0
        train_class_accs = {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in range(num_classes)}

        # ── Validate ──────────────────────────────────────────
        val_loss, val_acc, val_class_accs = validate(model, val_loader, criterion, device, num_classes)

        epoch_time = time.time() - epoch_start

        # ── Print ─────────────────────────────────────────────
        print(
            f"Epoch {epoch:3d}/{epochs} ({epoch_time:.1f}s) | "
            f"Train loss: {train_loss_avg:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
            f"Val acc_3: {val_class_accs[3]:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print(
            f"  Train per-class: {' '.join(f'{train_class_accs[c]:.3f}' for c in range(num_classes))} | "
            f"Val per-class: {' '.join(f'{val_class_accs[c]:.3f}' for c in range(num_classes))}"
        )

        # ── Wandb logging ─────────────────────────────────────
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss_avg, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for c in range(num_classes):
                log_dict[f"train/acc_{c}"] = train_class_accs[c]
                log_dict[f"val/acc_{c}"] = val_class_accs[c]
            wandb.log(log_dict)

        # ── Checkpointing (save in Lightning-compatible format) ──
        val_acc3 = val_class_accs[3]
        if val_acc3 > best_val_acc3:
            best_val_acc3 = val_acc3
            best_epoch = epoch
            patience_counter = 0
            save_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, save_path)
            print(f"  ★ New best val/acc_3: {val_acc3:.4f} → saved to {save_path}")
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            print(f"Best val/acc_3: {best_val_acc3:.4f} at epoch {best_epoch}")
            break

    # ── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training complete! Best val/acc_3: {best_val_acc3:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {os.path.join(ckpt_dir, ckpt_name)}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --raw --clf_weights {os.path.join(ckpt_dir, ckpt_name)}")
    print("=" * 60)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train triple-riding classifier (plain PyTorch)")
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG, help="Path to cfg.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="GPU device id")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="triple-riding-clf")
    parser.add_argument("--wandb_name", type=str, default="main_1_scratch_weighted_puf")
    args = parser.parse_args()
    train(args)
