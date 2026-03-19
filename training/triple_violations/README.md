# Triple Riding Classifier Training

Training script for the triple riding violation classifier based on the DashCop paper.

## Architecture

Based on the DashCop paper:
- **Backbone**: Frozen SAC module (YOLOv8 segmentation mask coefficient head)
- **Classifier**: Learnable convolutional layers + linear layer
- **Input**: R-M instance ROI crops (224x224 RGB images)
- **Output**: 4 classes (0=none, 1=single, 2=double, 3=triple riding)

## Dataset Structure

The dataset should be organized in one of these formats:

### Option 1: Class-folder structure (recommended)
```
datasets/
├── train/
│   ├── 0/          # or 0_none
│   ├── 1/          # or 1_single
│   ├── 2/          # or 2_double
│   └── 3/          # or 3_triple
└── val/
    ├── 0/
    ├── 1/
    ├── 2/
    └── 3/
```

### Option 2: YOLO-style structure
```
datasets/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt   # Contains: class_id (0-3)
│       └── ...
└── val/
    ├── images/
    └── labels/
```

## Configuration

Edit `cfg.yaml` to configure:
- `path`: Dataset root directory
- `sac_weights`: Path to SAC model weights (model_ft.pt)
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `lr`: Learning rate
- `device`: GPU device id or 'cpu'

## Usage

### Training
Use the PyTorch Lightning script to train the classifier. It reads configurations from `cfg.yaml` by default.
```bash
python train_lightning.py                        # Uses defaults from cfg.yaml
python train_lightning.py --cfg custom_cfg.yaml  # Use a custom config file
python train_lightning.py --resume /path/to/ckpt # Resume training from a checkpoint
python train_lightning.py --wandb                # Enable Weights & Biases logging
```

### Evaluation
Use `evaluate.py` to compute metrics (precision, recall, F1) and generate confusion matrices (4-class and binary).
```bash
# For newly trained models (raw argmax):
python evaluate.py --raw --clf_weights /path/to/new_model.ckpt

# For older models (uses YOLOClf +1 mapping):
python evaluate.py --clf_weights /path/to/tr_clf.ckpt
```

### Testing Dataset
```bash
python dataset.py cfg.yaml
```

## Output

Training produces:
- `checkpoints/tr_clf.ckpt` - Final model checkpoint
- `checkpoints/last.ckpt` - Last epoch checkpoint
- `checkpoints/logs/` - Training logs (CSV format)

## Using Trained Model

The checkpoint can be used with the inference pipeline:
```python
from pipeline_comp.yolo_clf import YOLOClf

clf = YOLOClf(
    weights_path='/path/to/tr_clf.ckpt',
    rm_assoc_path='/path/to/model_ft.pt'
)

# Classify an instance crop
prediction = clf(instance_crop)  # Returns: 0, 1, 2, or 3
```

## Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | none | No rider / invalid crop |
| 1 | single | Single rider |
| 2 | double | Two riders |
| 3 | triple | Three or more riders (violation) |
