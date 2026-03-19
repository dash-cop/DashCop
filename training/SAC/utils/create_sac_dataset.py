"""
SAC Dataset Preparation Script
Creates YOLO-style segmentation dataset with association IDs for the SAC model.

Dataset Split:
- Train: videoset2, videoset3, first 50 videos of videoset4
- Val: second 50 videos of videoset4  
- Test: videoset1

Annotation Format (SAC/CAS format):
<cls> <assoc_id> <xn1> <yn1> <xn2> <yn2> ...

Classes:
- 0: rider
- 1: motorcycle

Association: rider.association_id == motorcycle.motor_track_id
"""

import xml.etree.ElementTree as ET
import cv2
import os
import glob
import argparse
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import shutil


# === CONFIGURATION ===
ANNOTATIONS_BASE = "/ssd_scratch/sai.teja/annotations_backup_cvat_22Dec25"
VIDEOS_BASE = "/ssd_scratch/sai.teja/RideSafe_Dataset_WSD"
OUTPUT_BASE = "/ssd_scratch/sai.teja/SAC_dataset"

# Number of parallel workers
NUM_WORKERS = 8


def get_video_path(xml_path, videoset_name):
    """Get video path from XML path."""
    video_name = os.path.basename(xml_path).replace(".xml", ".mp4")
    return os.path.join(VIDEOS_BASE, videoset_name, "original_videos", video_name)


def parse_points_string(points_str):
    """Parse CVAT polyline points string to list of (x, y) tuples."""
    points = []
    for pt in points_str.split(';'):
        x, y = pt.split(',')
        points.append((float(x), float(y)))
    return points


def normalize_points(points, img_width, img_height):
    """Normalize points to [0, 1] range."""
    return [(x / img_width, y / img_height) for x, y in points]


def format_annotation_line(cls_id, assoc_id, normalized_points):
    """Format a single annotation line in SAC format."""
    coords = []
    for x, y in normalized_points:
        coords.extend([f"{x:.7f}", f"{y:.7f}"])
    return f"{cls_id} {assoc_id} " + " ".join(coords)


def extract_frame_annotations(xml_path, videoset_name):
    """
    Extract all frame annotations from an XML file.
    
    Returns:
        dict: {frame_num: {'riders': [...], 'motorcycles': [...], 'img_dims': (w, h)}}
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return {}
    
    # Get image dimensions from XML
    original_size = root.find('.//original_size')
    if original_size is not None:
        img_width = int(original_size.find('width').text)
        img_height = int(original_size.find('height').text)
    else:
        # Default dimensions if not found
        img_width = 2560
        img_height = 1440
    
    frame_data = defaultdict(lambda: {
        'riders': [],
        'motorcycles': [],
        'img_dims': (img_width, img_height)
    })
    
    # Process all tracks
    for track in root.findall('.//track'):
        label = track.attrib.get('label', '')
        
        if label == 'rider':
            # Only use polyline annotations for riders (segmentation model)
            for polyline in track.findall('polyline'):
                if polyline.attrib.get('outside') == "1":
                    continue
                    
                frame_num = int(polyline.attrib['frame'])
                
                # Get association_id
                assoc_id = -1
                for attr in polyline.findall('attribute'):
                    if attr.get('name') == 'association_id':
                        try:
                            assoc_id = int(attr.text)
                        except (ValueError, TypeError):
                            assoc_id = -1
                        break
                
                if assoc_id == -1:
                    continue
                
                # Extract polygon points
                points_str = polyline.attrib.get('points', '')
                if not points_str:
                    continue
                polygon = parse_points_string(points_str)
                
                frame_data[frame_num]['riders'].append({
                    'assoc_id': assoc_id,
                    'polygon': polygon
                })
        
        elif label == 'motorcycle':
            # Only use polyline annotations for motorcycles (segmentation model)
            for polyline in track.findall('polyline'):
                if polyline.attrib.get('outside') == "1":
                    continue
                    
                frame_num = int(polyline.attrib['frame'])
                
                # Get motor_track_id
                motor_id = -1
                for attr in polyline.findall('attribute'):
                    if attr.get('name') == 'motor_track_id':
                        try:
                            motor_id = int(attr.text)
                        except (ValueError, TypeError):
                            motor_id = -1
                        break
                
                if motor_id == -1:
                    continue
                
                # Extract polygon points
                points_str = polyline.attrib.get('points', '')
                if not points_str:
                    continue
                polygon = parse_points_string(points_str)
                
                frame_data[frame_num]['motorcycles'].append({
                    'assoc_id': motor_id,
                    'polygon': polygon
                })
    
    return dict(frame_data)


def process_video(args):
    """
    Process a single video: extract frames and create annotations.
    
    Args:
        args: tuple of (xml_path, videoset_name, output_dir)
    
    Returns:
        dict: {'saved': count, 'skipped': count}
    """
    xml_path, videoset_name, output_dir = args
    
    video_path = get_video_path(xml_path, videoset_name)
    video_name = os.path.basename(xml_path).replace('.xml', '')
    
    # Extract annotations from XML
    frame_data = extract_frame_annotations(xml_path, videoset_name)
    
    if not frame_data:
        return {'saved': 0, 'skipped': 0, 'no_data': 1}
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return {'saved': 0, 'skipped': 0, 'no_video': 1}
    
    saved = 0
    skipped = 0
    
    # Process each frame with annotations
    for frame_num, data in frame_data.items():
        riders = data['riders']
        motorcycles = data['motorcycles']
        img_width, img_height = data['img_dims']
        
        # Keep all riders and motorcycles regardless of correspondence
        if not riders and not motorcycles:
            skipped += 1
            continue
        
        # Read frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            skipped += 1
            continue
        
        # Create annotation lines
        annotation_lines = []
        
        # Add riders (cls=0)
        for rider in riders:
            norm_points = normalize_points(rider['polygon'], img_width, img_height)
            line = format_annotation_line(0, rider['assoc_id'], norm_points)
            annotation_lines.append(line)
        
        # Add motorcycles (cls=1)
        for moto in motorcycles:
            norm_points = normalize_points(moto['polygon'], img_width, img_height)
            line = format_annotation_line(1, moto['assoc_id'], norm_points)
            annotation_lines.append(line)
        
        # Save image and annotation
        base_name = f"{videoset_name}_{video_name}_frame{frame_num:06d}"
        img_path = os.path.join(output_dir, f"{base_name}.jpg")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        cv2.imwrite(img_path, frame)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(annotation_lines))
        
        saved += 1
    
    cap.release()
    return {'saved': saved, 'skipped': skipped}


def get_xml_files_for_split(split):
    """
    Get XML files for a specific split.
    
    Split definitions:
    - train: videoset2, videoset3, first 50 of videoset4
    - val: second 50 of videoset4
    - test: videoset1
    """
    xml_files = []
    
    if split == 'train':
        # All of videoset2
        vs2_files = sorted(glob.glob(os.path.join(ANNOTATIONS_BASE, "videoset2", "*.xml")))
        xml_files.extend([(f, "videoset2") for f in vs2_files])
        
        # All of videoset3
        vs3_files = sorted(glob.glob(os.path.join(ANNOTATIONS_BASE, "videoset3", "*.xml")))
        xml_files.extend([(f, "videoset3") for f in vs3_files])
        
        # First 50 of videoset4
        vs4_files = sorted(glob.glob(os.path.join(ANNOTATIONS_BASE, "videoset4", "*.xml")))[:50]
        xml_files.extend([(f, "videoset4") for f in vs4_files])
        
    elif split == 'val':
        # Second 50 of videoset4
        vs4_files = sorted(glob.glob(os.path.join(ANNOTATIONS_BASE, "videoset4", "*.xml")))[50:]
        xml_files.extend([(f, "videoset4") for f in vs4_files])
        
    elif split == 'test':
        # All of videoset1
        vs1_files = sorted(glob.glob(os.path.join(ANNOTATIONS_BASE, "videoset1", "*.xml")))
        xml_files.extend([(f, "videoset1") for f in vs1_files])
    
    return xml_files


def main():
    parser = argparse.ArgumentParser(description='Create SAC dataset from CVAT XML annotations')
    parser.add_argument('--output', type=str, default=OUTPUT_BASE,
                        help=f'Output directory (default: {OUTPUT_BASE})')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Which splits to process (default: all)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of parallel workers (default: {NUM_WORKERS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually processing')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAC Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Splits to process: {args.splits}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    
    # Process each split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")
        
        # Get XML files for this split
        xml_files = get_xml_files_for_split(split)
        print(f"Found {len(xml_files)} videos for {split}")
        
        if args.dry_run:
            print("DRY RUN - showing first 5 files:")
            for f, vs in xml_files[:5]:
                print(f"  {vs}: {os.path.basename(f)}")
            continue
        
        # Create output directory
        output_dir = os.path.join(args.output, split)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare arguments for parallel processing
        process_args = [(xml_path, vs_name, output_dir) 
                        for xml_path, vs_name in xml_files]
        
        # Process videos in parallel
        total_saved = 0
        total_skipped = 0
        
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_video, process_args),
                total=len(process_args),
                desc=f"Processing {split}"
            ))
        
        for result in results:
            total_saved += result.get('saved', 0)
            total_skipped += result.get('skipped', 0)
        
        print(f"\n{split} split complete:")
        print(f"  Frames saved: {total_saved}")
        print(f"  Frames skipped: {total_skipped}")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
