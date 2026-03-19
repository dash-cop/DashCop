"""
Balance Triple Riding Dataset

Creates a balanced dataset where all classes have equal representation.
The goal is triple riding detection, so we balance based on the minority class (class 3).

Strategy:
- Count triple riding instances (class 3) - the minority class
- For each split (train/val):
  - Sample (num_triple + extra) instances from classes 0, 1, 2
  - Keep all triple instances
  - Copy to new balanced dataset

Usage:
    python balance_crops.py --input_dir /path/to/tr_dataset --output_dir /path/to/bal_tr_dataset --extra 1500
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def count_images_in_class(class_dir):
    """Count number of images in a class directory."""
    if not os.path.exists(class_dir):
        return 0
    
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return len(image_files)


def get_class_distribution(dataset_root, split):
    """Get distribution of classes in a split."""
    split_dir = os.path.join(dataset_root, split)
    
    class_counts = {}
    class_names = ['0_none', '1_single', '2_double', '3_triple']
    
    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        count = count_images_in_class(class_dir)
        class_idx = int(class_name[0])
        class_counts[class_idx] = {
            'name': class_name,
            'count': count,
            'dir': class_dir
        }
    
    return class_counts


def sample_images(class_dir, num_samples, seed=42):
    """Randomly sample images from a class directory."""
    if not os.path.exists(class_dir):
        return []
    
    # Get all image files
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample
    random.seed(seed)
    if num_samples >= len(image_files):
        # If requesting more than available, return all
        return image_files
    else:
        return random.sample(image_files, num_samples)


def balance_dataset(input_dir, output_dir, extra_samples=1500, seed=42):
    """
    Create balanced dataset.
    
    Args:
        input_dir: Path to original dataset
        output_dir: Path to balanced dataset output
        extra_samples: Extra samples to add beyond minority class count
        seed: Random seed for reproducibility
    """
    
    print("="*60)
    print("Dataset Balancing for Triple Riding Detection")
    print("="*60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Extra samples per class: {extra_samples}")
    print(f"Random seed: {seed}")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both splits
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Get class distribution
        class_counts = get_class_distribution(input_dir, split)
        
        # Print original distribution
        total_original = sum(c['count'] for c in class_counts.values())
        print(f"\nOriginal distribution ({total_original} total):")
        for idx in sorted(class_counts.keys()):
            info = class_counts[idx]
            pct = 100.0 * info['count'] / total_original if total_original > 0 else 0
            print(f"  {info['name']:12s}: {info['count']:6d} ({pct:5.1f}%)")
        
        # Find minority class (should be class 3 - triple)
        minority_class = min(class_counts.keys(), key=lambda k: class_counts[k]['count'])
        minority_count = class_counts[minority_class]['count']
        
        print(f"\nMinority class: {class_counts[minority_class]['name']} ({minority_count} instances)")
        
        # Calculate target count for each class
        target_count = minority_count + extra_samples
        print(f"Target count per class: {target_count}")
        
        # Create output directories for this split
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process each class
        total_balanced = 0
        for class_idx in sorted(class_counts.keys()):
            info = class_counts[class_idx]
            class_name = info['name']
            class_dir = info['dir']
            available_count = info['count']
            
            # Create output class directory
            output_class_dir = os.path.join(split_output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Determine how many to sample
            if class_idx == minority_class:
                # Keep all minority class instances
                num_to_sample = available_count
            else:
                # Sample target_count from other classes
                num_to_sample = min(target_count, available_count)
            
            print(f"\n{class_name}: sampling {num_to_sample} from {available_count} available")
            
            # Sample images
            sampled_images = sample_images(class_dir, num_to_sample, seed=seed)
            
            # Copy images
            for img_file in tqdm(sampled_images, desc=f"Copying {class_name}"):
                src = os.path.join(class_dir, img_file)
                dst = os.path.join(output_class_dir, img_file)
                shutil.copy2(src, dst)
            
            total_balanced += len(sampled_images)
            print(f"  Copied {len(sampled_images)} images")
        
        # Print balanced distribution
        print(f"\n{'='*60}")
        print(f"Balanced distribution ({total_balanced} total):")
        balanced_counts = get_class_distribution(output_dir, split)
        for idx in sorted(balanced_counts.keys()):
            info = balanced_counts[idx]
            pct = 100.0 * info['count'] / total_balanced if total_balanced > 0 else 0
            print(f"  {info['name']:12s}: {info['count']:6d} ({pct:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("Balancing complete!")
    print(f"Balanced dataset saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Balance triple riding dataset for better training'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/ssd_scratch/sai.teja/tr_dataset_ogsplit',
        help='Path to original imbalanced dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/ssd_scratch/sai.teja/bal_tr_dataset_ogsplit',
        help='Path to save balanced dataset'
    )
    parser.add_argument(
        '--extra',
        type=int,
        default=1500,
        help='Extra samples to add beyond minority class count (default: 1500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Check if train and val splits exist
    for split in ['train', 'val']:
        split_dir = os.path.join(args.input_dir, split)
        if not os.path.exists(split_dir):
            print(f"Error: {split} split not found in {args.input_dir}")
            return
    
    # Run balancing
    balance_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        extra_samples=args.extra,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
