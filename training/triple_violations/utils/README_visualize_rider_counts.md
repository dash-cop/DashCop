# Rider Count Visualization Tool

Script to visualize the number of riders per motorcycle from CVAT XML annotations.

## Features

- Parses XML annotations to find motorcycles and their associated riders
- Counts riders per motorcycle using `motor_track_id` and `association_id` mapping
- Displays rider count on motorcycles in **RED BOLD** text
- Optional bounding box drawing for riders and motorcycles
- Saves annotated frames to output directory

## Usage

### Basic Usage
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --output ./output
```

### Draw Rider Bounding Boxes
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --draw-riders
```

### Don't Draw Motorcycle Bounding Boxes
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --no-draw-motors
```

### Draw Both Rider and Motor Boxes
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --draw-riders --draw-motors
```

### Save All Frames (not just annotated)
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --save-all-frames
```

### Change Frame Interval
```bash
python visualize_rider_counts.py --xml /path/to/annotation.xml --frame-interval 10
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--xml` | Path to XML annotation file (required) | - |
| `--output` | Output directory for frames | `./rider_count_visualization` |
| `--draw-riders` | Draw bounding boxes for riders | `False` |
| `--draw-motors` | Draw bounding boxes for motorcycles | `True` |
| `--no-draw-motors` | Don't draw motorcycle bboxes | - |
| `--save-all-frames` | Save all frames (not just annotated) | `False` |
| `--frame-interval` | Save every Nth frame | `5` |

## Output

The script creates a directory structure:
```
output_dir/
└── video_name/
    ├── frame_000000.jpg
    ├── frame_000005.jpg
    ├── frame_000010.jpg
    └── ...
```

Each frame shows:
- **Red number** on motorcycles indicating rider count
- **Yellow** bounding boxes for motorcycles (if `--draw-motors`)
- **Magenta** bounding boxes for riders (if `--draw-riders`)
- **Green** motor track ID label below motorcycle bbox

## Example

```bash
cd /home/sai.teja/DashCop/training/SAC/triple_violations/utils

python visualize_rider_counts.py \
    --xml /ssd_scratch/sai.teja/dataset_backup_cvat_22Dec25/videoset1/20211109123408_0060.xml \
    --output /ssd_scratch/sai.teja/rider_count_viz \
    --draw-riders \
    --draw-motors
```

This will:
1. Parse the XML file
2. Find the corresponding video
3. Extract motorcycle and rider annotations
4. Count riders per motorcycle
5. Save annotated frames showing rider counts

## Statistics

After processing, the script prints statistics like:
```
Rider count statistics (motorcycle instances):
  0 rider(s): 15 motorcycles
  1 rider(s): 234 motorcycles
  2 rider(s): 45 motorcycles
  3 rider(s): 12 motorcycles
```
