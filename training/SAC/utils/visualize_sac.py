"""
Visualize SAC dataset annotations.

Usage:
    python visualize_sac.py <image_path> <txt_path> [output_path]
    
Example:
    python visualize_sac.py datasets/train/images/2567-175.jpg datasets/train/labels/2567-175.txt output.jpg
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# Colors: BGR format
COLORS = {
    0: (0, 255, 0),    # rider: green
    1: (255, 0, 0),    # motorcycle: blue
}

CLASS_NAMES = {
    0: 'R',   # rider
    1: 'M',   # motorcycle
}


def parse_annotation_line(line):
    """
    Parse a single SAC annotation line.
    Format: <cls> <assoc_id> <x1> <y1> <x2> <y2> ...
    
    Returns:
        tuple: (cls_id, assoc_id, points) where points is list of (x, y) normalized coords
    """
    parts = line.strip().split()
    if len(parts) < 6:  # need at least cls, assoc_id, and 2 points
        return None
    
    cls_id = int(parts[0])
    assoc_id = int(parts[1])
    
    # Parse coordinate pairs
    coords = [float(x) for x in parts[2:]]
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    
    return cls_id, assoc_id, points


def denormalize_points(points, img_width, img_height):
    """Convert normalized points to pixel coordinates."""
    return [(int(x * img_width), int(y * img_height)) for x, y in points]


def get_centroid(points):
    """Calculate centroid of polygon points."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return int(np.mean(x_coords)), int(np.mean(y_coords))


def visualize_annotations(image_path, txt_path, output_path=None, text_scale=0.6):
    """
    Visualize SAC annotations on an image.
    
    Args:
        image_path: Path to the image file
        txt_path: Path to the YOLO-style txt annotation file
        output_path: Optional path to save the output image
        text_scale: Font scale for labels (default: 0.6)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Read annotations
    txt_path = Path(txt_path)
    if not txt_path.exists():
        print(f"Error: Annotation file not found: {txt_path}")
        return
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Draw annotations
    for line in lines:
        parsed = parse_annotation_line(line)
        if parsed is None:
            continue
        
        cls_id, assoc_id, norm_points = parsed
        
        # Denormalize points
        points = denormalize_points(norm_points, img_width, img_height)
        
        # Get color for this class
        color = COLORS.get(cls_id, (255, 255, 255))
        
        # Draw polyline (closed polygon)
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        
        # Fill polygon with transparency
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Draw centroid with assoc_id
        cx, cy = get_centroid(points)
        label = f"{CLASS_NAMES.get(cls_id, '?')}{assoc_id}"
        
        # Draw text background
        thickness = max(1, int(text_scale * 3))
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
        cv2.rectangle(img, (cx - text_w//2 - 2, cy - text_h//2 - 2), 
                      (cx + text_w//2 + 2, cy + text_h//2 + 2), color, -1)
        
        # Draw text
        cv2.putText(img, label, (cx - text_w//2, cy + text_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness)
    
    # Save output
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Saved visualization to {output_path}")
    
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Visualize SAC dataset annotations')
    parser.add_argument('--image_path',default = "/home/sai.teja/DashCop/training/SAC/datasets/train/2567-175.jpg",help='Path to the image file')
    parser.add_argument('--txt_path',default = "/home/sai.teja/DashCop/training/SAC/datasets/train/2567-175.txt", help='Path to the YOLO txt annotation file')
    parser.add_argument('--output',default = "output.jpg", help='Path to save output image')
    parser.add_argument('--text-scale', type=float, default=0.5, help='Font scale for labels (default: 0.6)')
    
    args = parser.parse_args()
    
    visualize_annotations(
        args.image_path, 
        args.txt_path, 
        output_path=args.output,
        text_scale=args.text_scale,
    )


if __name__ == '__main__':
    main()
