import os
import cv2 

is_idd_clips = True
all_folders = os.listdir('./idd_clips_xml')
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


# # clear alll files in hnh_frame/train/images and hnh_frame/train/labels
# for file in os.listdir('hnh_frame/train/images'):
#     os.remove('hnh_frame/train/images/' + file)

# for file in os.listdir('hnh_frame/train/labels'):
#     os.remove('hnh_frame/train/labels/' + file)

#     # clear alll files in hnh_frame/test/images and hnh_frame/test/labels
# for file in os.listdir('hnh_frame/test/images'):
#     os.remove('hnh_frame/test/images/' + file)

# for file in os.listdir('hnh_frame/test/labels'):
#     os.remove('hnh_frame/test/labels/' + file)

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

    # print(frame_wise_riders)
    # print(frame_wise_motor)

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
total_helmet = 0
total_no_helmet = 0
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

        # print(idd_folder)
        # print(video_folder)
        # videos = os.listdir(video_folder
        # video_name = videos[0]
        # print(video_name)
        # if video_name != '2745.mp4':
        #     continue
        video_name = task_name.split('.')[0]
        print("Processing video:", video_path)
        # annotation is in the folder/annotation.xml
        annotation_file_name = idd_folder + '/annotations.xml'

    else:
        print("Starting folder:", task_name)
        print("Left to process:", len(task_names) - i - 1)

        # print(idd_folder)
        # print(video_folder)
        # videos = os.listdir(video_folder
        # video_name = videos[0]
        # print(video_name)
        # if video_name != '2745.mp4':
        #     continue
        video_name = task_name.split('/')[-1] + '.mp4'
        print("Processing video:", video_name)
        # annotation is in the folder/annotation.xml
        annotation_file_name = task_name + '/annotations.xml'


    
    root = ET.parse(annotation_file_name).getroot()

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

        frame_width, frame_height = frame.shape[1], frame.shape[0]
        # print(frame_number, frame_width, frame_height)
        # print(frame_width, frame_height)

        # print(frame_number, frame_width, frame_height)

                # write the mototrcycle annotations
            

        out_annotation_file = open('hnh_frame/' + train_or_test + '/labels/' + video_name + '_' + str(frame_number) + '.txt', 'w')
    
        helmets = 0
        no_helmets = 0

        for track in root.findall('track'):
            # get label attribute
            label = track.get('label')

            if label == 'helmet' or label == 'no_helmet':

                for box in track.findall('box'):
                    box_frame = int(box.get('frame'))
                    if box_frame == frame_number and box.get('outside') == '0' and box.get('occluded') == '0':
                        # print("yes")
                        color = (0, 255, 0)
                        label_num = 0

                        coords = [float(box.get('xtl')), float(box.get('ytl')), float(box.get('xbr')), float(box.get('ybr'))]
                        if label == 'helmet':
                            label_num = 0
                            helmets += 1
                        else:
                            label_num = 1
                            color = (0, 0, 255)
                            no_helmets += 1

                        # convert coords from [xmin, ymin, xmax, ymax] to yolo format
                        x_center = (coords[0] + coords[2]) / 2 / frame_width
                        y_center = (coords[1] + coords[3]) / 2 / frame_height
                        width = (coords[2] - coords[0]) / frame_width
                        height = (coords[3] - coords[1]) / frame_height

                        # annotate the frame
                        xmin = x_center - width / 2
                        ymin = y_center - height / 2
                        xmax = x_center + width / 2
                        ymax = y_center + height / 2

                        xmin = max(0, xmin) * frame_width
                        ymin = max(0, ymin) * frame_height
                        xmax = min(1, xmax) * frame_width
                        ymax = min(1, ymax) * frame_height


                        # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                        out_annotation_file.write(str(label_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')

        out_annotation_file.close()

        
        # save the frame as video_name_frame_number.jpg file in hnh/train/images

        cv2.imwrite('hnh_frame/' + train_or_test + '/images/' + video_name + '_' + str(frame_number) + '.jpg', frame)
        if helmets == 0 and no_helmets == 0:
            # delete the annotation file if no helmets or no_helmets
            os.remove('hnh_frame/' + train_or_test + '/labels/' + video_name + '_' + str(frame_number) + '.txt')
            # remove the image file if no helmets or no_helmets
            os.remove('hnh_frame/' + train_or_test + '/images/' + video_name + '_' + str(frame_number) + '.jpg')
            # save frame as video_name_frame_number.jpg file in hnh/train/bg
            # cv2.imwrite('hnh/train/bg/' + video_name[:-4] + '_' + str(frame_number) + '.jpg', frame)

        total_helmet += helmets
        total_no_helmet += no_helmets

        frame_number += 1

    cap.release()

print('Total helmets:', total_helmet)
print('Total no helmets:', total_no_helmet)