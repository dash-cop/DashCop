"""
Data Preparation Script for Triple Riding Classifier

This script extracts R-M (Rider-Motorcycle) instance crops from video annotations
and organizes them into class folders for training.

Structure:
    datasets/
    ├── train/           # videoset 2, 3, 4(first 50 videos)
    │   ├── 0_none/      # No riders (just motorcycle)
    │   ├── 1_single/    # Single rider
    │   ├── 2_double/    # Two riders
    │   └── 3_triple/    # Three or more riders
    └── val/             # videoset 1
        ├── 0_none/
        ├── 1_single/
        ├── 2_double/
        └── 3_triple/

R-M Instance:
    - For each motorcycle (motor_track_id), find all riders with matching association_id
    - Merge all bounding boxes (motorcycle + riders) into one combined bbox
    - Crop that region from the frame
    - Save in folder based on rider count

Usage:
    python prepare_rm_crops.py --output ./datasets

    python prepare_rm_crops.py 2>&1 | tee tr_dataset.log
"""

import json
import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict


# === CONFIGURATION ===
BASE_VIDEO_DIR = "/data3/deepti.rawat/Wrong-side-violation/ridesafe_visualize/RideSafe_Dataset_WSD"
BASE_ANNOTATION_DIR = "/archive/deepti.rawat/Wrong-side-violation/annotations_backup_cvat_10Feb26"

# Videoset mapping for train/val split
TRAIN_VIDEOSETS = ["videoset2", "videoset3", "videoset4"]
VAL_VIDEOSETS = ["videoset1"]

# Class names for folder organization
CLASS_NAMES = {
    0: "0_none",      # No riders (just motorcycle)
    1: "1_single",    # Single rider
    2: "2_double",    # Two riders
    3: "3_triple"     # Three or more riders (violation)
}

# Minimum area threshold (relative to image area) to filter out too small instances
MIN_RELATIVE_AREA = 0.0025
MAX_RELATIVE_AREA = 10 # Filter out unreasonably large crops

# Anomaly JSON — frames in here are SKIPPED during crop extraction
ANOMALY_PATH = "/home/sai.teja/Dashcop_wsd/io_clf/utils/anomaly.json"

# Maps JSON videoset keys (v1, v2 ...) → actual videoset folder names
VIDEOSET_KEY_MAP = {
    "v1": "videoset1",
    "v2": "videoset2",
    "v3": "videoset3",
    "v4": "videoset4",
    "v5": "videoset5",
}


def get_video_path(xml_path):
    """Get video path from XML file path."""
    videoset = xml_path.split("/")[-2]
    video_name = xml_path.split("/")[-1].replace(".xml", ".mp4")
    return f"{BASE_VIDEO_DIR}/{videoset}/original_videos/{video_name}"


def load_anomaly_frames(json_path=ANOMALY_PATH):
    """Load anomaly frame list and return a nested dict for fast lookup.

    Returns:
        dict: {videoset_folder_name: {video_name: set(frame_numbers)}}
        e.g. {"videoset1": {"20211125115152_0060": {1255, 1270}, ...}, ...}
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] Anomaly JSON not found: {json_path} — no frames will be skipped.")
        return {}

    with open(json_path, "r") as f:
        raw = json.load(f)

    anomalies = {}
    for vkey, videos in raw.items():
        videoset = VIDEOSET_KEY_MAP.get(vkey)
        if videoset is None:
            print(f"[WARNING] Unknown videoset key '{vkey}' in anomaly JSON — skipping.")
            continue
        anomalies[videoset] = {
            video_name: set(frames)
            for video_name, frames in videos.items()
        }

    total_frames = sum(
        len(frames)
        for videos in anomalies.values()
        for frames in videos.values()
    )
    print(f"Loaded anomaly JSON: {total_frames} anomalous frames across "
          f"{sum(len(v) for v in anomalies.values())} videos.")
    return anomalies


def get_xml_files_for_videoset(videoset):
    """Get all XML files for a videoset."""
    return glob.glob(f"{BASE_ANNOTATION_DIR}/{videoset}/*.xml")


def parse_xml_for_rm_instances(xml_path):
    """Parse XML file and extract R-M instance data per frame.
    
    Returns:
        dict: {frame_num: {motor_track_id: {
            'motor_bbox': bbox dict,
            'rider_bboxes': [list of rider bboxes],
            'rider_count': int,
            'combined_bbox': merged bbox dict
        }}}
        img_dims: (width, height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return {}, (2560, 1440)
    
    # Get image dimensions
    original_size = root.find('.//original_size')
    if original_size is None:
        img_width, img_height = 2560, 1440
    else:
        img_width = int(original_size.find('width').text)
        img_height = int(original_size.find('height').text)
    
    # Store data per frame
    frame_data = defaultdict(lambda: {
        'motorcycles': {},  # motor_track_id -> bbox
        'riders': defaultdict(list)  # association_id -> [rider bboxes]
    })
    
    # Parse all tracks
    for track in root.findall('.//track'):
        label = track.attrib.get('label')
        
        for box in track.findall('box'):
            frame_number = int(box.attrib['frame'])
            
            # Skip outside boxes
            if box.attrib.get('outside') == "1":
                continue
            
            bbox = {
                'xtl': float(box.attrib['xtl']),
                'ytl': float(box.attrib['ytl']),
                'xbr': float(box.attrib['xbr']),
                'ybr': float(box.attrib['ybr'])
            }
            
            if label == 'motorcycle':
                # Get motor_track_id from motorcycle
                motor_track_id = None
                for attr in box.findall('attribute'):
                    if attr.get('name') == 'motor_track_id':
                        try:
                            motor_track_id = int(attr.text)
                        except (ValueError, TypeError):
                            motor_track_id = None
                        break
                
                if motor_track_id is not None and motor_track_id != -1: #ignore the None and -1 ids
                    frame_data[frame_number]['motorcycles'][motor_track_id] = bbox
            
            elif label == 'rider':
                # Get association_id (which links to motor_track_id)
                association_id = None
                for attr in box.findall('attribute'):
                    if attr.get('name') == 'association_id':
                        try:
                            association_id = int(attr.text)
                        except (ValueError, TypeError):
                            association_id = None
                        break
                
                if association_id is not None and association_id != -1: #ignore the None and -1 ids
                    frame_data[frame_number]['riders'][association_id].append(bbox)
    
    # Process data to create R-M instances with combined bboxes
    result = {}
    for frame_num, data in frame_data.items():
        result[frame_num] = {}
        
        for motor_track_id, motor_bbox in data['motorcycles'].items():
            # Find riders with matching association_id
            rider_bboxes = data['riders'].get(motor_track_id, [])
            rider_count = len(rider_bboxes)
            
            # Create combined bbox (motorcycle + all riders)
            combined_bbox = motor_bbox.copy()
            for rider_bbox in rider_bboxes:
                combined_bbox = {
                    'xtl': min(combined_bbox['xtl'], rider_bbox['xtl']),
                    'ytl': min(combined_bbox['ytl'], rider_bbox['ytl']),
                    'xbr': max(combined_bbox['xbr'], rider_bbox['xbr']),
                    'ybr': max(combined_bbox['ybr'], rider_bbox['ybr'])
                }
            
            result[frame_num][motor_track_id] = {
                'motor_bbox': motor_bbox,
                'rider_bboxes': rider_bboxes,
                'rider_count': rider_count,
                'combined_bbox': combined_bbox
            }
    
    return result, (img_width, img_height)


def crop_rm_instance(frame, bbox, img_dims, padding=10):
    """Crop R-M instance from frame with optional padding.
    
    Args:
        frame: CV2 image frame
        bbox: Bounding box dict with xtl, ytl, xbr, ybr
        img_dims: (width, height) of the image
        padding: Pixels to add around the crop
    
    Returns:
        Cropped image or None if invalid
    """
    img_width, img_height = img_dims
    
    # Get bbox coordinates with padding
    x1 = max(0, int(bbox['xtl']) - padding)
    y1 = max(0, int(bbox['ytl']) - padding)
    x2 = min(img_width, int(bbox['xbr']) + padding)
    y2 = min(img_height, int(bbox['ybr']) + padding)
    
    # Validate dimensions
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Check relative area
    crop_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    relative_area = crop_area / img_area
    
    if relative_area < MIN_RELATIVE_AREA or relative_area > MAX_RELATIVE_AREA:
        return None
    
    # Crop
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    return crop


def get_class_from_rider_count(rider_count):
    """Map rider count to class index.
    
    0: No riders (just motorcycle)
    1: Single rider
    2: Two riders
    3: Three or more riders
    """
    if rider_count <= 0:
        return 0
    elif rider_count == 1:
        return 1
    elif rider_count == 2:
        return 2
    else:  # 3 or more
        return 3


def process_video(xml_path, output_base_dir, split, stats=None, anomaly_frames=None):
    """Process one video and save R-M instance crops.

    Args:
        xml_path: Path to XML annotation file
        output_base_dir: Base output directory
        split: 'train' or 'val'
        stats: Dictionary to accumulate statistics
        anomaly_frames: nested dict from load_anomaly_frames();
                        frames in here are skipped.

    Returns:
        Number of crops saved
    """
    # Parse XML
    frame_data_all, img_dims = parse_xml_for_rm_instances(xml_path)
    
    if not frame_data_all:
        return 0
    
    # Get video path
    video_path = get_video_path(xml_path)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return 0
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0
    
    video_name = os.path.basename(xml_path).replace('.xml', '')
    saved_count = 0
    skipped_anomaly = 0

    # Build per-video anomaly set for quick lookup (empty set if no anomalies)
    videoset = xml_path.split("/")[-2]  # e.g. "videoset2"
    anomaly_set = set()
    if anomaly_frames:
        anomaly_set = anomaly_frames.get(videoset, {}).get(video_name, set())
    if anomaly_set:
        print(f"  [{video_name}] Will skip {len(anomaly_set)} anomalous frames: {sorted(anomaly_set)}")

    # Get all frames that have annotations (sorted)
    annotated_frames = sorted(frame_data_all.keys())

    # Process only frames with annotations
    for frame_idx in annotated_frames:
        # Skip anomalous frames
        if frame_idx in anomaly_set:
            skipped_anomaly += 1
            continue
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_data = frame_data_all[frame_idx]
        
        for motor_id, instance_data in frame_data.items():
            rider_count = instance_data['rider_count']
            combined_bbox = instance_data['combined_bbox']
            
            # Get class and folder name
            class_idx = get_class_from_rider_count(rider_count)
            class_folder = CLASS_NAMES[class_idx]
            
            # Crop R-M instance
            crop = crop_rm_instance(frame, combined_bbox, img_dims)
            
            if crop is None:
                continue
            
            # Create output path
            output_dir = os.path.join(output_base_dir, split, class_folder)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save crop with unique filename
            crop_filename = f"{video_name}_frame{frame_idx:06d}_motor{motor_id}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            saved_count += 1
            
            # Update statistics
            if stats is not None:
                stats[split][class_idx] += 1
    
    cap.release()
    if skipped_anomaly:
        print(f"  [{video_name}] Skipped {skipped_anomaly} anomalous frame(s).")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='Prepare R-M instance crops for triple riding classifier')
    parser.add_argument('--output', type=str, default='/ssd_scratch/sai.teja/tr_dataset_ogsplit', help='Output directory for datasets')
    parser.add_argument('--anomaly_json', type=str, default=ANOMALY_PATH,
                        help='Path to anomaly JSON (frames to skip). Pass empty string to disable.')

    args = parser.parse_args()

    output_dir = args.output

    print("=" * 70)
    print("R-M Instance Crop Preparation for Triple Riding Classifier")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Processing ALL annotated frames from XML files")
    print(f"\nTrain videosets: {TRAIN_VIDEOSETS}")
    print(f"Val videosets: {VAL_VIDEOSETS}")
    print(f"\nClass folders:")
    for idx, name in CLASS_NAMES.items():
        print(f"  {idx}: {name}")
    print("=" * 70)

    # Load anomaly frames to skip
    anomaly_frames = load_anomaly_frames(args.anomaly_json) if args.anomaly_json else {}
    
    # Create output directories
    for split in ['train', 'val']:
        for class_name in CLASS_NAMES.values():
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Statistics
    stats = {
        'train': {0: 0, 1: 0, 2: 0, 3: 0},
        'val': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    # Process training set
    print("\n" + "=" * 70)
    print("Processing TRAINING set")
    print("=" * 70)
    
    for videoset in TRAIN_VIDEOSETS:
        xml_files = sorted(get_xml_files_for_videoset(videoset))
        if videoset == "videoset4":
            xml_files = xml_files[:50]
            print(f"\n{videoset}: using first 50 videos (out of {len(get_xml_files_for_videoset(videoset))} total)")
        else:
            print(f"\n{videoset}: {len(xml_files)} videos")

        for xml_path in tqdm(xml_files, desc=f"Processing {videoset}"):
            process_video(xml_path, output_dir, 'train', stats, anomaly_frames)

    # Process validation set
    print("\n" + "=" * 70)
    print("Processing VALIDATION set")
    print("=" * 70)

    for videoset in VAL_VIDEOSETS:
        xml_files = get_xml_files_for_videoset(videoset)
        print(f"\n{videoset}: {len(xml_files)} videos")

        for xml_path in tqdm(xml_files, desc=f"Processing {videoset}"):
            process_video(xml_path, output_dir, 'val', stats, anomaly_frames)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    print("\nTraining set:")
    train_total = 0
    for class_idx, count in stats['train'].items():
        print(f"  {CLASS_NAMES[class_idx]}: {count:,} crops")
        train_total += count
    print(f"  TOTAL: {train_total:,} crops")
    
    print("\nValidation set:")
    val_total = 0
    for class_idx, count in stats['val'].items():
        print(f"  {CLASS_NAMES[class_idx]}: {count:,} crops")
        val_total += count
    print(f"  TOTAL: {val_total:,} crops")
    
    print("\n" + "=" * 70)
    print(f"Output directory structure:")
    print(f"{output_dir}/")
    print(f"├── train/")
    for name in CLASS_NAMES.values():
        print(f"│   ├── {name}/")
    print(f"└── val/")
    for name in CLASS_NAMES.values():
        print(f"    ├── {name}/")
    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
