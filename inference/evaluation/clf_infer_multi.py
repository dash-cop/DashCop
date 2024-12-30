import torch
import torch.multiprocessing as mp
import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
import os
from instance_funcs import *
from models.load import *
from models.lit_model import Model

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

out_annot_folder = "/ssd_scratch/cvit/keshav/instance_crops_clf_pred/"
out_annot_folder = "/ssd_scratch/cvit/Video_set_1/instance_crops_clf_pred/"
os.makedirs(out_annot_folder, exist_ok=True)

def give_model(rank):
    model = give_yolo_model(num_classes=4)
    model = Model(model, None, 4)
    # model_to_load = torch.load("/ssd_scratch/cvit/keshav/keshav/sdc/logs/lightning_logs/version_65/checkpoints/last.ckpt")
    # model_to_load = torch.load("/ssd_scratch/cvit/keshav/last.ckpt")
    model_to_load = torch.load("tr_clf.ckpt")
    model.load_state_dict(model_to_load['state_dict'])
    model.to(f"cuda:{rank}")
    model.eval()
    return model
    
# Function to be run by each process
def worker(rank, model, pred_files):

    # for idx, file in enumerate(pred_files):
    #     print(f"RANK : {rank} ", file)
    #     vid_name = file.split("/")[-1].split(".")[0]
    #     # video_path = "/ssd_scratch/cvit/keshav/test_videos/" + f"{vid_name}.mp4"
    #     video_path = "/ssd_scratch/cvit/Video_set_1/videos/" + f"{vid_name}.mp4"
    #     cap = cv2.VideoCapture(video_path)
    #     all_frames = []
    #     for frame_number in range(0, 2000, 5):  # Assuming frames start from 0 and increment by 5
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if(frame_number % 5 != 0):
    #             continue
    #         all_frames.append(frame)
    #     # Initialize dictionary to store motorcycle and rider information
    #     frame_data = {}
    #     all_data = []
    #     all_inf_data = {}
    #     with open(file, "r") as f:
    #         all_lines = f.readlines()
    #         for line in all_lines:
    #             line = list(map(int, line.strip().split()))
    #             fn, label, l, t, r, b, tid = line
    #             if(fn not in all_inf_data):
    #                 all_inf_data[fn] = [[label, l, t, r, b, tid]]
    #             else:
    #                 all_inf_data[fn].append([label, l, t, r, b, tid])
    #             inst_crop = cv2.resize(all_frames[fn//5][t:b, l:r] / 255, (224, 224))
    #             # inst_crop_ = (inst_crop - mean[None, None]) / std[None, None]
    #             crop = torch.from_numpy(inst_crop.astype(np.float32)).permute(2, 0, 1)[None].to(f"cuda:{rank}")
    #             out_pred = torch.argmax(model(crop)[0]).item() + 1
    #             if(out_pred == 4):
    #                 out_pred = 0
    #             all_data.append([fn, out_pred, l, t, r, b, tid])
    #             # cv2.imwrite(f"crops/{vid_name}_{fn}_{tid}_{out_pred}.png", inst_crop*255)
        
    #     file_name = out_annot_folder + f"/{vid_name}.txt"
    #     with open(file_name, "w") as f:
    #         for inst in all_data:
    #             line = " ".join(map(str, inst))
    #             f.write(line + "\n")
    for idx, file in enumerate(pred_files):
        print(file)
        vid_name = file.split("/")[-1].split(".")[0]
        video_path = "/ssd_scratch/cvit/Video_set_1/videos/" + f"{vid_name}.mp4"
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        for frame_number in range(0, 2000, 5):  # Assuming frames start from 0 and increment by 5
            ret, frame = cap.read()
            if not ret:
                break
            if(frame_number % 5 != 0):
                continue
            all_frames.append(frame)
        # Initialize dictionary to store motorcycle and rider information
        print("Length of all_frames", len(all_frames))
        if len(all_frames) == 0:
            continue
        frame_data = {}
        all_data = []
        all_inf_data = {}
        with open(file, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                print(file)
                print(line)
                fn, label, l, t, r, b, tid,lp_num = line.strip().split()
                fn, label, l, t, r, b, tid,lp_num = int(fn), int(label), int(l), int(t), int(r), int(b), int(tid), lp_num
                if(fn not in all_inf_data):
                    all_inf_data[fn] = [[label, l, t, r, b, tid,lp_num]]
                else:
                    all_inf_data[fn].append([label, l, t, r, b, tid, lp_num])
                print(fn//5, t, b, l, r)
                inst_crop = cv2.resize(all_frames[fn//5][t:b, l:r] / 255, (224, 224))
                # inst_crop_ = (inst_crop - mean[None, None]) / std[None, None]
                crop = torch.from_numpy(inst_crop.astype(np.float32)).permute(2, 0, 1)[None].to(f"cuda:{rank}")
                out_pred = torch.argmax(model(crop)[0]).item() + 1
                if(out_pred == 4):
                    out_pred = 0
                all_data.append([fn, out_pred, l, t, r, b, tid, lp_num])
                # cv2.imwrite(f"crops/{vid_name}_{fn}_{tid}_{out_pred}.png", inst_crop*255)
        
        file_name = out_annot_folder + f"/{vid_name}.txt"
        with open(file_name, "w") as f:
            for inst in all_data:
                line = " ".join(map(str, inst))
                f.write(line + "\n")

    # exit(0)


# Main function to set up multiprocessing
def main():
    # Create a list to keep track of processes
    processes = []
    pred_files = glob.glob("/ssd_scratch/cvit/keshav/instance_crops_pred/*.txt")
    pred_files = glob.glob("/ssd_scratch/cvit/Video_set_1/instance_crops_pred_new/*.txt")

    # Start 4 processes
    for rank in range(4):
        batch = len(pred_files) // 4
        files = pred_files[rank*batch:(rank+1)*batch]
        model = give_model(rank)
        p = mp.Process(target=worker, args=(rank, model, files))
        p.start()
        processes.append(p)

    # Join all processes
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Necessary for some platforms
    main()
