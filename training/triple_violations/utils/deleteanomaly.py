"""
Delete anomaly frame images from the triple riding dataset.

This script reads anomaly frames from a JSON file and deletes matching
images from the dataset folders.

Usage:
    python deleteanomaly.py
"""

import json
import os
import glob

# Path to anomaly JSON
ANOMALY_PATH = "/home/sai.teja/Dashcop_wsd/io_clf/utils/anomaly.json"

# Dataset base directory
DATASET_DIR = "/ssd_scratch/sai.teja/triple_violaions/tr_dataset"

# Mapping from JSON keys to split folders
# v1 -> videoset1 -> val (since videoset1 is used for validation)
# v2, v3, v4 -> videoset2,3,4 -> train
VIDEOSET_TO_SPLIT = {
    "v1": "val",
    "v2": "train",
    "v3": "train",
    "v4": "train",
    "v5": "train"
}

# Class folders
CLASS_FOLDERS = ["0_none", "1_single", "2_double", "3_triple"]


def load_anomaly_json(json_path):
    """Load anomaly frames from JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: Anomaly JSON not found: {json_path}")
        return {}
    
    with open(json_path, 'r') as f:
        return json.load(f)


def delete_anomaly_images(anomalies, dataset_dir):
    """Delete images matching anomaly video names and frame numbers.
    
    Image naming format: {video_name}_frame{frame_num:06d}_motor{motor_id}.jpg
    """
    total_deleted = 0
    
    for videoset_key, videos in anomalies.items():
        # Get the split folder (train or val)
        split = VIDEOSET_TO_SPLIT.get(videoset_key)
        if split is None:
            print(f"Unknown videoset key: {videoset_key}, skipping...")
            continue
        
        print(f"\nProcessing {videoset_key} ({split}):")
        
        for video_name, frame_numbers in videos.items():
            # Convert frame numbers to set for faster lookup
            frame_set = set(frame_numbers)
            
            # Search in all class folders
            for class_folder in CLASS_FOLDERS:
                folder_path = os.path.join(dataset_dir, split, class_folder)
                
                if not os.path.exists(folder_path):
                    continue
                
                # Find all images matching this video name
                pattern = os.path.join(folder_path, f"{video_name}_frame*.jpg")
                matching_files = glob.glob(pattern)
                
                for file_path in matching_files:
                    filename = os.path.basename(file_path)
                    
                    # Extract frame number from filename
                    # Format: {video_name}_frame{frame_num:06d}_motor{motor_id}.jpg
                    try:
                        # Split by '_frame' and then by '_motor'
                        frame_part = filename.split("_frame")[1]
                        frame_num = int(frame_part.split("_motor")[0])
                        
                        if frame_num in frame_set:
                            os.remove(file_path)
                            print(f"  Deleted: {filename}")
                            total_deleted += 1
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Could not parse frame from {filename}: {e}")
                        continue
    
    return total_deleted


def main():
    print("=" * 60)
    print("Delete Anomaly Frame Images")
    print("=" * 60)
    print(f"Anomaly JSON: {ANOMALY_PATH}")
    print(f"Dataset directory: {DATASET_DIR}")
    print("=" * 60)
    
    # Load anomaly data
    anomalies = load_anomaly_json(ANOMALY_PATH)
    
    if not anomalies:
        print("No anomaly data found or JSON is empty.")
        return
    
    # Show what will be deleted
    print("\nAnomalies to delete:")
    for videoset_key, videos in anomalies.items():
        split = VIDEOSET_TO_SPLIT.get(videoset_key, "unknown")
        print(f"  {videoset_key} ({split}):")
        for video_name, frames in videos.items():
            print(f"    {video_name}: {len(frames)} frames")
    
    # Delete images
    print("\n" + "-" * 60)
    print("Deleting images...")
    print("-" * 60)
    
    total_deleted = delete_anomaly_images(anomalies, DATASET_DIR)
    
    print("\n" + "=" * 60)
    print(f"DONE! Total images deleted: {total_deleted}")
    print("=" * 60)


if __name__ == "__main__":
    main()