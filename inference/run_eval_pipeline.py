import subprocess
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

vids_1 = glob.glob("/nas/deepti.rawat/Wrong-side-driving/Videos/videoset1/original_videos/*.mp4")
vids_4 = glob.glob("/nas/deepti.rawat/Wrong-side-driving/Videos/videoset4/original_videos/*.mp4")[50:] # Only process the last 50 videos from vids_4

vids = vids_1

NUM_GPUS = 3

program_path = "pipeline.py"

def run_command(vid, gpu_id):
    """
    Run the command for a specific video on a specific GPU
    
    :param vid: Path to the video file
    :param gpu_id: GPU device number to use
    :return: Tuple of (video path, success status, error message if any)
    """
    vid_name = vid.split("/")[-1]
    
    # Skip if already processed
    if os.path.exists('/ssd_scratch/sai.teja/vidset1_masks/' + vid_name.split(".")[0] + ".npzq"):
        print(f"Skipping (already processed): {vid_name}")
        return (vid, True, None)
    
    try:
        # Prepare the command with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        command = ["python3", program_path, "--video", vid]
        
        print(f"Executing command on GPU {gpu_id}: {' '.join(command)}")
        
        # Run the command with the specific GPU (output will be shown in real-time)
        result = subprocess.run(command, check=True, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                                )
        
        print(f"Successfully processed on GPU {gpu_id}: {vid_name}")
        return (vid, True, None)
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running command for {vid} on GPU {gpu_id}: {e}\n"
        error_msg += f"STDOUT: {e.stdout}\n"
        error_msg += f"STDERR: {e.stderr}"
        return (vid, False, error_msg)


def process_videos_in_parallel(videos, num_gpus=4):
    """
    Process videos in parallel across specified number of GPUs

    :param videos: List of video paths
    :param num_gpus: Number of GPUs to use
    """
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # Create futures for each video, cycling through GPU IDs
        futures = {
            executor.submit(run_command, vid, gpu_id % num_gpus): vid 
            for gpu_id, vid in enumerate(videos)
        }

        # Track and log results
        successful_videos = []
        failed_videos = []

        for future in as_completed(futures):
            vid, success, error = future.result()

            if success:
                successful_videos.append(vid)
            else:
                failed_videos.append(vid)
                print(error)

        # Print summary
        print("\n--- Processing Summary ---")
        print(f"Total videos: {len(videos)}")
        print(f"Successfully processed: {len(successful_videos)}")
        print(f"Failed videos: {len(failed_videos)}")

        if failed_videos:
            print("\nFailed Videos:")
            for vid in failed_videos:
                print(vid)

            # Write failed videos to a file
            with open("failed_videos.txt", "w") as f:
                for vid in failed_videos:
                    f.write(f"{vid}\n")

            print("Failed video list saved to failed_videos.txt")


# Run with 4 GPUs by default (adjust num_gpus as needed)
process_videos_in_parallel(vids, num_gpus=NUM_GPUS)
