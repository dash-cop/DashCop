import xml.etree.ElementTree as ET
import glob
import cv2
import os
from tqdm import tqdm

xml_files = glob.glob("/nas/deepti.rawat/Wrong-side-driving/Annotations/annotations_backup_cvat_10Feb26/videoset1/*.xml")
# video_files = ["/ssd_scratch/cvit/saiteja/RideSafe_Dataset_WSD/videoset1/original_videos/" + f.split("/")[-1].replace(".xml", ".mp4") for f in xml_files]
out_annot_folder = "/ssd_scratch/sai.teja/triple_riding/videoset1/instance_crops_gt/"
os.makedirs(out_annot_folder, exist_ok=True)

#chekc if xml files and video files exist

if len(xml_files) == 0 :
    print("No XML or video files found. Please check the paths.")
    exit(1)

# Load the XML file
for i, file in enumerate(tqdm(xml_files, desc="Processing XML files")):
    vid_name = file.split("/")[-1].replace(".xml", "")
    tree = ET.parse(file)
    root = tree.getroot()

    # Initialize dictionary to store motorcycle and rider information
    frame_data = {}
    # cap = cv2.VideoCapture(video_files[i])
    # print(video_files[i])
    all_frames = []
    all_data = []
    # Iterate through each frame
    for frame_number in range(0, 2000):  # Assuming frames start from 0 and increment by 5
        # ret, frame = cap.read()
        # all_frames.append(frame)
        # if not ret:
            # break
        if(frame_number % 5 != 0):
            continue
        motorcycle_info = []
        rider_info = []

        # Find motorcycle and riders for the current frame
        for track in root.findall('.//track'):
            for box in track.findall('box'):
                if int(box.attrib['frame']) == frame_number:
                    if(box.attrib['outside'] == "1"):
                        continue
                    if track.attrib['label'] == 'motorcycle':
                        # Get motor_track_id from attributes
                        motorcycle_id = -1
                        for attr in box.findall('attribute'):
                            if attr.get('name') == 'motor_track_id':
                                try:
                                    motorcycle_id = int(attr.text)
                                except (ValueError, TypeError):
                                    motorcycle_id = -1
                                break
                        
                        # Skip if ID is -1 or invalid
                        if motorcycle_id == -1:
                            continue
                        
                        bbox = {
                            'xtl': float(box.attrib['xtl']),
                            'ytl': float(box.attrib['ytl']),
                            'xbr': float(box.attrib['xbr']),
                            'ybr': float(box.attrib['ybr'])
                        }
                        motorcycle_info.append({'id': motorcycle_id, 'bbox': bbox})
                    elif track.attrib['label'] == 'rider':
                        # Get association_id from attributes
                        association_id = -1
                        for attr in box.findall('attribute'):
                            if attr.get('name') == 'association_id':
                                try:
                                    association_id = int(attr.text)
                                except (ValueError, TypeError):
                                    association_id = -1
                                break
                        
                        # Skip if ID is -1 or invalid
                        if association_id == -1:
                            continue
                        
                        bbox = {
                            'xtl': float(box.attrib['xtl']),
                            'ytl': float(box.attrib['ytl']),
                            'xbr': float(box.attrib['xbr']),
                            'ybr': float(box.attrib['ybr'])
                        }
                        rider_info.append({'id': association_id, 'bbox': bbox})

        # Store the motorcycle and rider information for the current frame
        frame_data[frame_number] = {'motorcycles': motorcycle_info, 'riders': rider_info}

    # Now, for each frame, print motorcycle and rider information
    for frame_num, data in frame_data.items():
        # print(f"Frame {frame_num}:")
        for motorcycle in data['motorcycles']:
            # print(f"  Motorcycle ID: {motorcycle['id']}, BBox: {motorcycle['bbox']}")
            comb_bbox = {'xtl' : 10000, 'ytl' : 10000, 'xbr' : -1, 'ybr' : -1}
            combined_box = {
                    'xtl': min(motorcycle['bbox']['xtl'], comb_bbox['xtl']),
                    'ytl': min(motorcycle['bbox']['ytl'], comb_bbox['ytl']),
                    'xbr': max(motorcycle['bbox']['xbr'], comb_bbox['xbr']),
                    'ybr': max(motorcycle['bbox']['ybr'], comb_bbox['ybr'])
            }
            num_riders = 0
            for rider in data['riders']:
                if(rider['id'] != motorcycle['id']) or rider['id'] == -1 or motorcycle['id'] == -1:
                    continue
                num_riders += 1
                # print(f"  Rider Assoc ID: {rider['id']}, BBox: {rider['bbox']}")
                combined_box = {
                    'xtl': min(combined_box['xtl'], rider['bbox']['xtl']),
                    'ytl': min(combined_box['ytl'], rider['bbox']['ytl']),
                    'xbr': max(combined_box['xbr'], rider['bbox']['xbr']),
                    'ybr': max(combined_box['ybr'], rider['bbox']['ybr'])
                }
            # print(num_riders)
            label = num_riders
            track_id = motorcycle['id']
            l, t, r, b = int(combined_box['xtl']), int(combined_box['ytl']), int(combined_box['xbr']), int(combined_box['ybr'])
            all_data.append([frame_num, label, l, t, r, b, track_id])
    
    file_name = out_annot_folder + f"/{vid_name}.txt"
    with open(file_name, "w") as f:
        for inst in all_data:
            line = " ".join(map(str, inst))
            f.write(line + "\n")

            # if(num_riders >= 3):
            #     cv2.imwrite(f"{out_folder}/{vid_name}_{frame_num}_{motorcycle['id']}.png", all_frames[frame_num][t:b, l:r, :])
                # exit(0)