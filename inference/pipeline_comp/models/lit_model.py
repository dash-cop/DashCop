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
    def __init__(self, model, optimizer, num_classes):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.class_correct_train = [0 for _ in range(num_classes)]
        self.total_samples_train = [0 for _ in range(num_classes)]
        self.class_correct_val = [0 for _ in range(num_classes)]
        self.total_samples_val = [0 for _ in range(num_classes)]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # return logits.sum()
        loss = self.criterion(logits.sigmoid(), y)
        self.log('Train Loss', loss, batch_size=len(x), on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        for i in range(self.num_classes):
            cls_y = y[y == i]
            cls_logits = logits[y == i]
            self.class_correct_train[i] += (cls_logits.max(-1)[1] == cls_y).sum()
            self.total_samples_train[i] += len(cls_y)

        return loss
    
    def on_train_epoch_end(self):
        # all_corr = self.all_gather(self.class_correct).sum()
        # all_samples = self.all_gather(self.total_samples).sum()
        for i in range(self.num_classes):
            acc = self.class_correct_train[i] / self.total_samples_train[i]
            self.log(f"T{i}", acc, sync_dist=True, prog_bar=True)
            self.class_correct_train[i] = 0
            self.total_samples_train[i] = 0
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.sigmoid(), y)
        acc = (logits.max(-1)[1] == y).sum() / len(x)
        for i in range(self.num_classes):
            cls_y = y[y == i]
            cls_logits = logits[y == i]
            self.class_correct_val[i] += (cls_logits.max(-1)[1] == cls_y).sum()
            self.total_samples_val[i] += len(cls_y)
        return loss
    
    def on_validation_epoch_end(self):
        for i in range(self.num_classes):
            acc = self.class_correct_val[i] / self.total_samples_val[i]
            self.log(f"V{i}", acc, sync_dist=True, prog_bar=True)
            self.class_correct_val[i] = 0
            self.total_samples_val[i] = 0
    
    def configure_optimizers(self):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50], gamma=0.1)
        return [self.optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]