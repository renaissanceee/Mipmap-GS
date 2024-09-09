import subprocess
import argparse

def execute_x2(scene, root):
    command = (
        f"python pseudo_GT/SwinIR/main_test_swinir.py "
        f"--task classical_sr "
        f"--scale 2 "
        f"--training_patch_size 64 "
        f"--model_path pseudo_GT/SwinIR/model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth "
        f"--folder_lq {root}/{scene}/test/ours_30000/test_preds_8/ "
        f"--folder_gt {root}/{scene}/test/ours_30000/gt_4/ "
        f"--save_dir {root}/{scene}/pseudo_gt/swin_x4 "
        f"--dataset blender "
    )
    subprocess.run(command, shell=True)

def execute_x4(scene, root):
    command = (
        f"python pseudo_GT/SwinIR/main_test_swinir.py "
        f"--task classical_sr "
        f"--scale 4 "
        f"--training_patch_size 64 "
        f"--model_path pseudo_GT/SwinIR/model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth "
        f"--folder_lq {root}/{scene}/test/ours_30000/test_preds_8/ "
        f"--folder_gt {root}/{scene}/test/ours_30000/gt_2/ "
        f"--save_dir {root}/{scene}/pseudo_gt/swin_x2 "
        f"--dataset blender "
    )
    subprocess.run(command, shell=True)

def execute_x8(scene, root):
    command = (
        f"python pseudo_GT/SwinIR/main_test_swinir.py "
        f"--task classical_sr "
        f"--scale 8 "
        f"--training_patch_size 64 "
        f"--model_path pseudo_GT/SwinIR/model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth "
        f"--folder_lq {root}/{scene}/test/ours_30000/test_preds_8/ "
        f"--folder_gt {root}/{scene}/test/ours_30000/gt_1/ "
        f"--save_dir {root}/{scene}/pseudo_gt/swin_x1 "
        f"--dataset blender "
    )
    subprocess.run(command, shell=True)

def copy_json(scene, root, source):
    command = f"cp {source}/{scene}/transforms_test.json {root}/{scene}/pseudo_gt/swin_x1/"
    subprocess.run(command, shell=True)
    command = f"cp {source}/{scene}/transforms_test.json {root}/{scene}/pseudo_gt/swin_x2/"
    subprocess.run(command, shell=True)
    command = f"cp {source}/{scene}/transforms_test.json {root}/{scene}/pseudo_gt/swin_x4/"
    subprocess.run(command, shell=True)

def main(source, gt_root, scenes):
    for scene in scenes:
        execute_x2(scene, gt_root)
        execute_x4(scene, gt_root)
        execute_x8(scene, gt_root)
        copy_json(scene, gt_root, source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument('--source', type=str, default="../3D-Gaussian/nerf_synthetic", help="json")
    parser.add_argument('--gt_root', type=str, default="benchmark_nerf_synthetic_stmt_up",help="imgs")

    args = parser.parse_args()
    scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
    main(args.source, args.gt_root, scenes)