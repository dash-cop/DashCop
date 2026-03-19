"""
Training script for triple-riding classifier.

Trains the YOLO-based classifier (CustomLayer head) using PyTorch Lightning.
Saves checkpoints compatible with yolo_clf.py / evaluate.py.

Usage:
    python train_lightning.py                           # uses defaults from cfg.yaml
    python train_lightning.py --epochs 50 --lr 0.0005
    python train_lightning.py --resume /path/to/ckpt    # resume training


    python train_lightning.py 2>&1 | tee train_lightning.log
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from load import give_yolo_model
from lit_model import Model
from dataset import TripleRidingDataset


# ─── Defaults ────────────────────────────────────────────────
DEFAULT_CFG = "cfg.yaml"
# ─────────────────────────────────────────────────────────────


def load_config(cfg_path):
    """Load training config from YAML file."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg, resume_ckpt=None):
    """
    Build the model + optimizer in the exact same structure as yolo_clf.py.
    
    Model chain:
        give_yolo_model() → raw YOLO with CustomLayer classifier
        Model(model, optimizer, 4) → LightningModule wrapper
    
    The saved checkpoint will have keys matching this structure,
    so evaluate.py / yolo_clf.py can load it with:
        model.load_state_dict(ckpt['state_dict'])
    """
    sac_weights = cfg.get("sac_weights", "/archive/sai.teja/rm_assoc_model/model_ft.pt")
    num_classes = cfg.get("nc", 4)
    lr = cfg.get("lr", 0.001)
    
    # Build YOLO backbone + classifier head
    print(f"Building model from backbone: {sac_weights}")
    yolo_model = give_yolo_model(weights=sac_weights, num_classes=num_classes)

    # Handle freeze/unfreeze based on config
    unfreeze_all = cfg.get("unfreeze_all", False)
    freeze_all = cfg.get("freeze_all_except_classifier", False)

    if unfreeze_all:
        print("Unfreezing ALL layers for full fine-tuning")
        for param in yolo_model.parameters():
            param.requires_grad = True
    elif freeze_all:
        print("Freezing all layers except classifier")
        for name, param in yolo_model.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    # else: load.py already set the unfreeze policy (classifier, cv4, 15)

    # Print final freeze state
    print("\n" + "=" * 60)
    print("Final parameter freeze state:")
    print("=" * 60)
    for name, param in yolo_model.model.named_parameters():
        print(f"  {name} {param.shape} requires_grad={param.requires_grad}")
    print("=" * 60)


    # exit(0)
    # Create optimizer (only trainable params)
    trainable_params = [p for p in yolo_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters:     {sum(p.numel() for p in yolo_model.parameters()):,}")

    # Wrap in LightningModule
    model = Model(yolo_model, optimizer, num_classes, lr=lr)

    # Load pretrained classifier weights if resuming
    if resume_ckpt and os.path.isfile(resume_ckpt):
        print(f"Loading checkpoint for resume: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    return model


def build_dataloaders(cfg):
    """Build train and val dataloaders from config."""
    data_root = cfg.get("path", "/ssd_scratch/sai.teja/bal_tr_dataset")
    batch_size = cfg.get("batch_size", 32)
    imgsz = cfg.get("imgsz", 224)
    num_workers = cfg.get("num_workers", 4)
    use_weighted_sampling = cfg.get("weighted_sampling", False)

    train_dataset = TripleRidingDataset(data_root, split="train", imgsz=imgsz)
    val_dataset = TripleRidingDataset(data_root, split="val", imgsz=imgsz)

    # Weighted sampling: oversample minority classes using inverse frequency
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
        shuffle = False  # sampler and shuffle are mutually exclusive
        print(f"Weighted sampling enabled:")
        for i, (cnt, w) in enumerate(zip(class_counts, class_weights)):
            print(f"  Class {i}: count={cnt}, weight={w:.6f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def train(args):
    # ── Load config ───────────────────────────────────────────
    cfg = load_config(args.cfg)

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 10)
    ckpt_dir = cfg.get("checkpoint_dir", "/ssd_scratch/sai.teja/triple_violations/checkpoints")
    ckpt_name = cfg.get("checkpoint_name", "tr_clf")
    device_id = cfg.get("device", 0)
    if str(device_id) != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ── Build model ───────────────────────────────────────────
    model = build_model(cfg, resume_ckpt=args.resume)

    # ── Build dataloaders ─────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Wandb logger ──────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or ckpt_name,
            config=cfg,
            save_dir=ckpt_dir,
        )
        # Initialise the wandb run so we can call define_metric.
        # WandbLogger lazily inits; experiment property forces it.
        _ = wandb_logger.experiment
        # Use "epoch" as the x-axis for all metrics so charts align
        # with train_scratch.py when comparing runs in the same project.
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
    else:
        wandb_logger = None

    # Strip .ckpt if user included it in config name
    ckpt_base = ckpt_name.removesuffix(".ckpt")

    # ── Callbacks ─────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_base + "-{epoch:02d}",
        monitor="val/acc_3",        # monitor triple-class val accuracy
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/acc_3",
        patience=patience,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="cpu" if str(device_id) == "cpu" else "gpu",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        precision="32-true",
    )

    # ── Train ─────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # ── Save final checkpoint in compatible format ────────────
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt}")
        # Copy best checkpoint to a clean name for easy use
        final_path = os.path.join(ckpt_dir,ckpt_name)
        import shutil
        shutil.copy2(best_ckpt, final_path)
        print(f"Copied best model to: {final_path}")
    else:
        print("\nNo best checkpoint found (training may have been too short)")

    print("\nDone! To evaluate:")
    print(f"  python evaluate.py --raw --clf_weights {os.path.join(ckpt_dir, ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train triple-riding classifier")
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG,
                        help="Path to cfg.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="triple-riding-clf",
                        help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default="weigthed_default_puf",
                        help="Wandb run name (defaults to checkpoint_name from cfg)")
    args = parser.parse_args()
    train(args)
