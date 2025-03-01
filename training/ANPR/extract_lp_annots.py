import os
import cv2 

is_idd_clips = False
all_folders = os.listdir('./tr_clips_vidset2_extracted')
task_names = []
if is_idd_clips:

    # only folders with task in the name
    xml_folders = [folder for folder in all_folders if 'task' in folder]
    task_names = [folder.split('.')[0] for folder in xml_folders]
    # add .mp4 to task names
    task_names = [task + '.mp4' for task in task_names]
    # remove task_ from the task names
    task_names = [task.replace('task_', '') for task in task_names]
    # add test/ to the folder names
    xml_folders = ['./idd_clips_xml/' + folder for folder in xml_folders]
else:
    xml_folders = [folder for folder in all_folders]
    xml_folders = ['./tr_clips_vidset2_extracted/' + folder for folder in xml_folders]
    task_names = xml_folders

print(len(xml_folders))

if is_idd_clips:
    video_folders = os.listdir('./iddclips_videos')

    # add test/ to the folder names
    video_folders = ['./iddclips_videos/' + folder for folder in video_folders]
    print(video_folders)
else:
    video_names = os.listdir('./video_set_2_videos')

    print(video_names)


# clear alll files in license_plates/train/images and license_plates/train/labels
for file in os.listdir('license_plates/train/images'):
    os.remove('license_plates/train/images/' + file)

for file in os.listdir('license_plates/train/labels'):
    os.remove('license_plates/train/labels/' + file)

    # clear alll files in license_plates/test/images and license_plates/test/labels
for file in os.listdir('license_plates/test/images'):
    os.remove('license_plates/test/images/' + file)

for file in os.listdir('license_plates/test/labels'):
    os.remove('license_plates/test/labels/' + file)

def calculate_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = area1 + area2 - intersection
    return intersection / union

def get_rm_instances(root):
    # loop over the tracks in root
    frame_wise_riders = {} # key : frame, value : {track_id,xmin,ymin,xmax,ymax, associated_motorcycle_id(default = -1)}
    frame_wise_motor = {} # key : frame, value : {track_id,xmin,ymin,xmax,ymax}
    for track in root.findall('track'):
        # label is an attribute of track
        label = track.get('label')
        if label == 'rider' and track.find('box') is not None:
            rider_id = track.get('id')
            for box in track.findall('box'):
                frame = int(float(box.get('frame')))
                xmin = box.get('xtl')
                ymin = box.get('ytl')
                xmax = box.get('xbr')
                ymax = box.get('ybr')
                xmin,ymin,xmax,ymax = float(xmin),float(ymin),float(xmax),float(ymax)
                if box.get('outside') == '0' and box.get('occluded') == '0':
                    if frame not in frame_wise_riders:
                        frame_wise_riders[frame] = []
                    frame_wise_riders[frame].append({'track_id':rider_id, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'associated_motorcycle_id':0})

        if label == 'motorcycle' and track.find('box') is not None:
            motorcycle_id = track.get('id')
            for box in track.findall('box'):
                frame = int(float(box.get('frame')))
                xmin = box.get('xtl')
                ymin = box.get('ytl')
                xmax = box.get('xbr')
                ymax = box.get('ybr')
                xmin,ymin,xmax,ymax = float(xmin),float(ymin),float(xmax),float(ymax)
                if box.get('outside') == '0' and box.get('occluded') == '0':
                    if frame not in frame_wise_motor:
                        frame_wise_motor[frame] = []
                    frame_wise_motor[frame].append({'track_id':motorcycle_id, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax})

    # loop through all the frames which are keys in frame_wise_riders, then loop through all the riders in that frame and then loop through all the motorcycles in that frame and assign the id of the motorcycle as the associated_motorcycle_id of the rider with the highest IoU

    for key in frame_wise_riders.keys():
        riders = frame_wise_riders[key]
        if key not in frame_wise_motor.keys():
            continue
        motorcycles = frame_wise_motor[key]
        for rider in riders:
            max_iou = 0
            for motorcycle in motorcycles:
                iou = calculate_iou(rider, motorcycle)
                if iou > max_iou:
                    max_iou = iou
                    rider['associated_motorcycle_id'] = motorcycle['track_id']

    # for key,value in frame_wise_riders.items():
    #     if(len(value) > 1):
    #         print (key, value)


    return frame_wise_riders, frame_wise_motor




# import to read xml files
import xml.etree.ElementTree as ET
all_license_plates = 0
train_or_test = ''
for i, task_name in enumerate(task_names):
    
    if i%8 == 0:
        train_or_test = 'test'
    else:
        train_or_test = 'train'

    if is_idd_clips:
        # idd_folder is the name out of idd_folders which contains task_name in it
        for folder in xml_folders:
            if task_name in folder:
                idd_folder = folder
                break
        print(task_name)
        print(video_folders)
        # video is in the folder/data
        for folder in video_folders:
            if task_name in folder:
                video_path = folder
                break

        print("Starting folder:", idd_folder)
        print("Left to process:", len(xml_folders) - i - 1)

        video_name = task_name.split('.')[0]
        print("Processing video:", video_path)
        # annotation is in the folder/annotation.xml
        annotation_file_name = idd_folder + '/annotations.xml'

    else:
        print("Starting folder:", task_name)
        print("Left to process:", len(task_names) - i - 1)

        video_name = task_name.split('/')[-1] + '.mp4'
        print("Processing video:", video_name)
        # annotation is in the folder/annotation.xml
        annotation_file_name = task_name + '/annotations.xml'


    
    root = ET.parse(annotation_file_name).getroot()
    
    frame_wise_riders, frame_wise_motor = get_rm_instances(root)

    # read the video and loop through the frames
    if is_idd_clips:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture('./video_set_2_videos/' + video_name)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # every fifth frame
        if frame_number % 5 != 0:
            frame_number += 1
            continue

        # Check IF frame_number is a key in frame_wise_riders and frame_wise_motor
        if frame_number not in frame_wise_riders or frame_number not in frame_wise_motor:
            frame_number += 1
            continue

        frame_width, frame_height = frame.shape[1], frame.shape[0]

        motor_list = frame_wise_motor[frame_number]
        rider_list = frame_wise_riders[frame_number]
        for instance_num, motor_dict in enumerate(motor_list):

            riders_on_motor = [rider for rider in rider_list if rider['associated_motorcycle_id'] == motor_dict['track_id']]
            
            if len(riders_on_motor) > 0:
                # get the roi_xmin which is the minimum of all the xmins of the riders on the motorcycle and the xmin of the motorcycle
                roi_xmin = min(motor_dict['xmin'], min([rider['xmin'] for rider in riders_on_motor]))
                # get the roi_ymin which is the minimum of all the ymins of the riders on the motorcycle and the ymin of the motorcycle
                roi_ymin = min(motor_dict['ymin'], min([rider['ymin'] for rider in riders_on_motor]))
                # get the roi_xmax which is the maximum of all the xmaxs of the riders on the motorcycle and the xmax of the motorcycle
                roi_xmax = max(motor_dict['xmax'], max([rider['xmax'] for rider in riders_on_motor]))
                # get the roi_ymax which is the maximum of all the ymaxs of the riders on the motorcycle and the ymax of the motorcycle
                roi_ymax = max(motor_dict['ymax'], max([rider['ymax'] for rider in riders_on_motor]))

                # get the roi_width and roi_height
                roi_width = roi_xmax - roi_xmin
                roi_height = roi_ymax - roi_ymin

                # write the mototrcycle annotations
            

                out_annotation_file = open('license_plates/' + train_or_test + '/labels/' + video_name + '_' + str(frame_number) + '_' + str(instance_num) + '.txt', 'w')
                roi_frame = frame[int(roi_ymin):int(roi_ymax), int(roi_xmin):int(roi_xmax)]
                if roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0:
                    print("roi_frame shape is 0")
                    continue
                license_plates = 0

                for track in root.findall('track'):
                    # get label attribute
                    label = track.get('label')

                    if label == 'license_plate':

                        for box in track.findall('box'):
                            box_frame = int(box.get('frame'))
                            if box_frame == frame_number and box.get('outside') == '0' and box.get('occluded') == '0':
                                # print("yes")
                                color = (0, 255, 0)
                                label_num = 0

                                coords = [float(box.get('xtl')), float(box.get('ytl')), float(box.get('xbr')), float(box.get('ybr'))]

                                # check if the box is inside the roi
                                if coords[0] >= roi_xmin and coords[1] >= roi_ymin and coords[2] <= roi_xmax and coords[3] <= roi_ymax:

                                    # print("license plate found")
                                    license_plates += 1
                                    # adjust the coords to the roi
                                    coords[0] -= roi_xmin
                                    coords[1] -= roi_ymin
                                    coords[2] -= roi_xmin
                                    coords[3] -= roi_ymin

                                    # expand the coords by 10%
                                    width = coords[2] - coords[0]
                                    height = coords[3] - coords[1]
                                    coords[0] -= 0.05 * width
                                    coords[1] -= 0.05 * height
                                    coords[2] += 0.05 * width
                                    coords[3] += 0.05 * height

                                    coords[0] = max(0, coords[0])
                                    coords[1] = max(0, coords[1])
                                    coords[2] = min(roi_width, coords[2])
                                    coords[3] = min(roi_height, coords[3])

                                    # convert coords from [xmin, ymin, xmax, ymax] to yolo format
                                    x_center = (coords[0] + coords[2]) / 2 / roi_width
                                    y_center = (coords[1] + coords[3]) / 2 / roi_height
                                    width = (coords[2] - coords[0]) / roi_width
                                    height = (coords[3] - coords[1]) / roi_height

                                    # annotate the frame
                                    xmin = x_center - width / 2
                                    ymin = y_center - height / 2
                                    xmax = x_center + width / 2
                                    ymax = y_center + height / 2

                                    xmin = max(0, xmin) * roi_width
                                    ymin = max(0, ymin) * roi_height
                                    xmax = min(1, xmax) * roi_width
                                    ymax = min(1, ymax) * roi_height


                                    cv2.rectangle(roi_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                                    out_annotation_file.write(str(label_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')

                out_annotation_file.close()

                
                # save the frame as video_name_frame_number.jpg file in hnh/train/images

                cv2.imwrite('license_plates/' + train_or_test + '/images/' + video_name + '_' + str(frame_number) + '_' + str(instance_num) + '.jpg', roi_frame)
                if license_plates == 0:
                    # delete the annotation file if license plates are not present
                    os.remove('license_plates/' + train_or_test + '/labels/' + video_name + '_' + str(frame_number) + '_' + str(instance_num) + '.txt')
                    # remove the image file if license plates are not present
                    os.remove('license_plates/' + train_or_test + '/images/' + video_name + '_' + str(frame_number) + '_' + str(instance_num) + '.jpg')

                
                all_license_plates += license_plates

        frame_number += 1

    cap.release()

print("Total license plates:", all_license_plates)