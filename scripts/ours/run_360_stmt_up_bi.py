# single-scale training and multi-scale testing setting proposed in mip-splatting

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time


scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room", "flowers", "treehill"]
factors = [8] * len(scenes)

excluded_gpus = set([])

output_dir = "360v2_ours_stmt_up_bi"
gt_root = "benchmark_360v2_stmt_up"

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    trained_gaussian = os.path.join(gt_root, scene, "point_cloud/iteration_30000/point_cloud.ply")
    for scale in [4, 2, 1]:
        pseudo_gt = os.path.join(gt_root, scene, f"pseudo_gt/resize_x{scale}")
        gt_folder = os.path.join(gt_root, scene, f"test/ours_30000/gt_{scale}")
        model_path= os.path.join(output_dir,scene,f"resize_x{scale}")
        cmd = f"python train_mipmap.py -s {pseudo_gt} -m {model_path} -r 1 --port {2009 + int(gpu)} --load_gaussian {trained_gaussian}"
        print(cmd)
        if not dry_run:
            os.system(cmd)
            
        cmd = f"python render_ours.py -m {model_path} --scale {scale} -r 1 --iteration 1000 --gt_folder {gt_folder}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet
    print(jobs)
    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

