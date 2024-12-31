import subprocess
import glob
import os

vids = glob.glob("/ssd_scratch/cvit/keshav/vid1_1/*.mp4")

program_path = "pipeline.py"

# Loop through the argument list and run the program with each set of arguments
for vid in vids:
    vid_name = vid.split("/")[-1]
    if(os.path.exists('/ssd_scratch/cvit/keshav/vidset1_masks/' + vid_name.split(".")[0] + ".npzq")):
       print("Continuing " + '/ssd_scratch/cvit/keshav/vidset1_dets/' + vid_name.split(".")[0] + ".npz")
       continue
    command = ["python3", program_path, "--video", vid]#, "--assoc", "box"]
    # Run the command
    try:
        print("Executing command:", " ".join(command))
        subprocess.run(command, check=True)
        print("Command execution successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
