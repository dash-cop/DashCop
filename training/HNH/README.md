## Helmet & No Helmet Detection

This repository provides tools and instructions for detecting helmets and no helmets in images using YOLOv8.

### Model Training

We utilize [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for training the detection model. Please ensure you have the YOLOv8 codebase set up before proceeding.

### Data Preprocessing

The script for data preprocessing is included in this directory:
- **`extract_hnh_annots_frame.py`**: This script prepares the dataset by extracting and annotating frames for helmet and no helmet detection.

### Training Details

The detection model is trained on full-frame images for helmet and no helmet detection. Ensure the dataset is preprocessed using the provided script before starting the training.

### Pretrained Weights

Pretrained weights for the model can be downloaded from the following link:
- [Google Drive - Pretrained Weights](https://drive.google.com/drive/folders/1NWt42_Sr7jxuAVZq2nQOm0LiZmE_3nYz?usp=drive_link)

### Steps to Get Started
1. Clone the YOLOv8 repository:
   ```bash
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   pip install -r requirements.txt
   ```
2. Preprocess the dataset using `extract_hnh_annots_frame.py`.
3. Train the detection model using YOLOv8 with the preprocessed dataset.
4. Use the pretrained weights provided in the Google Drive link for inference or further fine-tuning.