## ALPR Training

This repository provides tools and instructions for Automatic License Plate Recognition (ALPR) training. The ALPR module consists of two main components:

1. **License Plate Detection**: Detecting license plates in RGB frames.
2. **License Plate Recognition**: Optical Character Recognition (OCR) of the license plate.

### License Plate Detection

For license plate detection, we use [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) as the detection framework.

#### Data Preprocessing
The dataset for training can be preprocessed using the provided script:
- **`extract_lp_annots.py`**: This script processes the dataset and generates annotations for license plate detection.

#### Training
1. Clone the YOLOv8 repository:
   ```bash
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   pip install -r requirements.txt
   ```
2. Use `extract_lp_annots.py` to preprocess the dataset.
3. Train the YOLOv8 model for license plate detection using the preprocessed dataset.

### License Plate Recognition (OCR)

For OCR, we refer to the following repository:
- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

Follow the instructions in the linked repository for setting up and training the OCR model.

### Steps to Get Started
1. Preprocess the dataset for license plate detection using `extract_lp_annots.py`.
2. Train the detection model using YOLOv8.
3. Set up and train the OCR model using the Deep Text Recognition Benchmark repository.
4. Integrate both components to complete the ALPR pipeline.