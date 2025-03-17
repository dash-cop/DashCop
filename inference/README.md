# Traffic Violation Detection Pipeline

This directory contains the code for inference and running the traffic violation detection pipeline. Below is an overview of the code structure and instructions to run the pipeline effectively.

## Directory Structure

- **`pipeline_comp/`**: Contains the code for different sections of the pipeline.
- **`tracker/`**: Contains the code for the association tracker.
- **`ultralytics/`**: Contains the code for SAC and other detection models derived from YOLOv8.
- **`pipeline.py`**: Main Python file that orchestrates the pipeline. It includes various flags to customize the pipeline execution.

## Download Pretrained Weights
You can download the pretrained weights for SAC, HHN, and the TR-Classifier here
https://drive.google.com/drive/folders/1NWt42_Sr7jxuAVZq2nQOm0LiZmE_3nYz?usp=drive_link

## Running the Pipeline

### Flags
The `pipeline.py` script uses flags for configuration. Below are the available flags:

- **`--video`**: Path to the input video. Use `0` for webcam input.
  - Default: `/ssd_scratch/cvit/keshav/videoset1/original_videos/20211109152409_0060.mp4`
- **`--output`**: Path to save the output video.
  - Default: `./outputs/trip_offline_assoc_.mp4`
- **`--frame_start`**: Start frame for processing.
  - Default: `0`
- **`--frame_max`**: Maximum frame to process. Use `-1` for all frames.
  - Default: `-1`
- **`--infer_lp`**: Whether to infer license plates or not.
  - Default: `False`
- **`--infer_hnh`**: Whether to infer helmet/no-helmet violations or not.
  - Default: `True`
- **`--clf`**: Whether to use classification or not.
  - Default: `True`
- **`--tracker_config`**: Path to the association tracker configuration file.
  - Default: `./tracker/assoc_tracker/base_cfg.yaml`
- **`--rm_weights`**: Path to the rider-motorcycle association weights.
  - Default: `/ssd_scratch/cvit/keshav/model_ft.pt`
- **`--hnh_weights`**: Path to the helmet/no-helmet weights.
  - Default: `/ssd_scratch/cvit/keshav/weights/best.pt`
- **`--clf_weights`**: Path to classification weights.
  - Default: `/ssd_scratch/cvit/keshav/tr_clf.ckpt`
- **`--lp_det_weights`**: Path to license plate detection weights.
  - Default: `/ssd_scratch/cvit/keshav/lisence_plate_v8.pt`
- **`--lp_rec_weights`**: Path to license plate recognition model weights (e.g., trocr).
  - Default: `/ssd_scratch/cvit/keshav/lp_ckpt`
- **`--dont_show`**: Whether to suppress video output display.
  - Default: `True`

### Steps to Run
1. **Download Pretrained Weights**: Ensure you download the necessary pretrained weights. Provide links in this section if available.

2. **Set Up the Environment**:
   - Install the required dependencies.
   - Verify the paths to weights and configuration files.

3. **Run the Pipeline**:
   ```bash
   python pipeline.py \
       --video /path/to/video.mp4 \
       --output ./output/video_result.mp4 \
       --frame_start 0 \
       --frame_max -1 \
       --infer_lp True \
       --infer_hnh True \
       --clf True \
       --tracker_config ./tracker_reid/offline_tracker/base_cfg.yaml \
       --rm_weights /path/to/model_ft.pt \
       --hnh_weights /path/to/best.pt \
       --clf_weights /path/to/tr_clf.ckpt \
       --lp_det_weights /path/to/license_plate_v8.pt \
       --lp_rec_weights /path/to/lp_ckpt \
       --dont_show True
   ```

4. **Customize Flags**: Adjust the flags as needed to suit your use case.

