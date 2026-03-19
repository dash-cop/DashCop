"""
Triple Riding Classification Dataset

Dataset class for loading R-M instance ROI crops in YOLO-style folder structure.

Expected folder structure:
datasets/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt  # Contains class label (0, 1, 2, or 3)
│       ├── img002.txt
│       └── ...
└── val/
    ├── images/
    └── labels/

OR class-folder structure:
datasets/
├── train/
│   ├── 0_none/
│   ├── 1_single/
│   ├── 2_double/
│   └── 3_triple/
└── val/
    ├── 0_none/
    ├── 1_single/
    ├── 2_double/
    └── 3_triple/
"""

import os
import glob
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TripleRidingDataset(Dataset):
    """
    Dataset for triple riding classification.
    Supports two data formats:
    1. YOLO-style: images/ and labels/ folders with txt files containing class id
    2. Class-folder style: images organized in class-named subdirectories
    """
    
    def __init__(self, data_root, split='train', transform=None, imgsz=224):
        """
        Args:
            data_root: Root directory containing train/val subdirectories
            split: 'train' or 'val'
            transform: Optional custom transforms
            imgsz: Image size for resizing (default 224)
        """
        self.data_root = data_root
        self.split = split
        self.imgsz = imgsz
        
        # Default transforms
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((imgsz, imgsz)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((imgsz, imgsz)),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
        
        # Load data paths and labels
        self.image_paths = []
        self.labels = []
        self._load_data()
        
        print(f"Loaded {len(self.image_paths)} samples for {split}")
        self._print_class_distribution()
    
    def _load_data(self):
        """Load image paths and labels from dataset directory."""
        split_dir = os.path.join(self.data_root, self.split)
        
        # Check for class-folder structure first
        class_folders = ['0_none', '1_single', '2_double', '3_triple']
        if any(os.path.isdir(os.path.join(split_dir, cf)) for cf in class_folders):
            self._load_class_folder_structure(split_dir)
        # Check for simple numeric class folders (0, 1, 2, 3)
        elif any(os.path.isdir(os.path.join(split_dir, str(i))) for i in range(4)):
            self._load_numeric_folder_structure(split_dir)
        # Otherwise try YOLO-style structure
        else:
            self._load_yolo_structure(split_dir)
    
    def _load_class_folder_structure(self, split_dir):
        """Load from class-named folder structure."""
        class_mapping = {
            '0_none': 0, 'none': 0,
            '1_single': 1, 'single': 1,
            '2_double': 2, 'double': 2,
            '3_triple': 3, 'triple': 3
        }
        
        for folder_name, class_id in class_mapping.items():
            class_dir = os.path.join(split_dir, folder_name)
            if os.path.isdir(class_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in glob.glob(os.path.join(class_dir, ext)):
                        self.image_paths.append(img_path)
                        self.labels.append(class_id)
    
    def _load_numeric_folder_structure(self, split_dir):
        """Load from numeric class folder structure (0, 1, 2, 3)."""
        for class_id in range(4):
            class_dir = os.path.join(split_dir, str(class_id))
            if os.path.isdir(class_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in glob.glob(os.path.join(class_dir, ext)):
                        self.image_paths.append(img_path)
                        self.labels.append(class_id)
    
    def _load_yolo_structure(self, split_dir):
        """Load from YOLO-style images/ and labels/ structure."""
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.isdir(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in glob.glob(os.path.join(images_dir, ext)):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, f"{img_name}.txt")
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        # Label file should contain just the class id (0, 1, 2, or 3)
                        # Or YOLO format: class_id x_center y_center width height
                        parts = content.split()
                        class_id = int(parts[0])
                        
                        self.image_paths.append(img_path)
                        self.labels.append(class_id)
    
    def _print_class_distribution(self):
        """Print class distribution for the dataset."""
        class_names = ['none', 'single', 'double', 'triple']
        labels_array = np.array(self.labels)
        
        print(f"Class distribution for {self.split}:")
        for i, name in enumerate(class_names):
            count = np.sum(labels_array == i)
            pct = 100.0 * count / len(self.labels) if len(self.labels) > 0 else 0
            print(f"  {name} ({i}): {count} ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(cfg_path, batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders from config.
    
    Args:
        cfg_path: Path to cfg.yaml
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_root = cfg.get('path', './datasets')
    imgsz = cfg.get('imgsz', 224)
    batch_size = cfg.get('batch_size', batch_size)
    
    train_dataset = TripleRidingDataset(data_root, split='train', imgsz=imgsz)
    val_dataset = TripleRidingDataset(data_root, split='val', imgsz=imgsz)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = "cfg.yaml"
    
    print(f"Testing dataset with config: {cfg_path}")
    train_loader, val_loader = get_dataloaders(cfg_path)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
