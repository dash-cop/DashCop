import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

class Model(L.LightningModule):
    def __init__(self, model, optimizer, num_classes, lr=0.001):
        super().__init__()
        self.automatic_optimization = False  # Manual optimization
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.class_correct_train = [0 for _ in range(num_classes)]
        self.total_samples_train = [0 for _ in range(num_classes)]
        self.class_correct_val = [0 for _ in range(num_classes)]
        self.total_samples_val = [0 for _ in range(num_classes)]
        self._debug_printed = False

    def on_train_epoch_start(self):
        """Explicitly set module modes at the start of each training epoch.
        
        Matches the configuration from diagnose_train.py which works:
        - All modules: train mode
        - Only FROZEN BN: eval mode (use pretrained running stats)
        - Trainable BN (cv4, layer 15): stay in train mode (batch stats)
        """
        # Force ALL modules to train mode
        for module in self.modules():
            module.training = True
        
        # Only put FROZEN BN layers in eval mode
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                all_frozen = all(not p.requires_grad for p in module.parameters())
                if all_frozen:
                    module.eval()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch

        # Debug on first step
        if not self._debug_printed:
            train_count = sum(1 for m in self.modules() if m.training)
            eval_count = sum(1 for m in self.modules() if not m.training)
            print(f"\n[DEBUG] Train modules: {train_count}, Eval modules: {eval_count}")
            print(f"[DEBUG] Manual optimization: {not self.automatic_optimization}")
            self._debug_printed = True

        # Manual forward/backward/step (same as plain PyTorch)
        opt.zero_grad()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.manual_backward(loss)
        opt.step()

        self.log('train/loss', loss, batch_size=len(x), on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        for i in range(self.num_classes):
            cls_y = y[y == i]
            cls_logits = logits[y == i]
            self.class_correct_train[i] += (cls_logits.max(-1)[1] == cls_y).sum().item()
            self.total_samples_train[i] += len(cls_y)
    
    def on_train_epoch_end(self):
        total_correct = 0
        total_samples = 0
        for i in range(self.num_classes):
            if self.total_samples_train[i] > 0:
                acc = self.class_correct_train[i] / self.total_samples_train[i]
                self.log(f"train/acc_{i}", acc, sync_dist=True, prog_bar=True)
                total_correct += self.class_correct_train[i]
                total_samples += self.total_samples_train[i]
            self.class_correct_train[i] = 0
            self.total_samples_train[i] = 0
        if total_samples > 0:
            self.log("train/acc", total_correct / total_samples, sync_dist=True, prog_bar=True)

        # Step the LR scheduler manually
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val/loss', loss, batch_size=len(x), on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        for i in range(self.num_classes):
            cls_y = y[y == i]
            cls_logits = logits[y == i]
            self.class_correct_val[i] += (cls_logits.max(-1)[1] == cls_y).sum().item()
            self.total_samples_val[i] += len(cls_y)
        return loss
    
    def on_validation_epoch_end(self):
        total_correct = 0
        total_samples = 0
        for i in range(self.num_classes):
            if self.total_samples_val[i] > 0:
                acc = self.class_correct_val[i] / self.total_samples_val[i]
                self.log(f"val/acc_{i}", acc, sync_dist=True, prog_bar=True)
                total_correct += self.class_correct_val[i]
                total_samples += self.total_samples_val[i]
            self.class_correct_val[i] = 0
            self.total_samples_val[i] = 0
        if total_samples > 0:
            self.log("val/acc", total_correct / total_samples, sync_dist=True, prog_bar=True)
    
    def configure_optimizers(self):
        # Create optimizer HERE (after Lightning has moved model to device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        print(f"[configure_optimizers] Created fresh Adam optimizer with lr={self.lr}")
        print(f"[configure_optimizers] Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"[configure_optimizers] First param device: {trainable_params[0].device}")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]