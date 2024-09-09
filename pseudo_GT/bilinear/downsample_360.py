import os
import cv2
import subprocess
import argparse


def main(source, gt_root, scenes):
    for scene in scenes:
        input_dir = os.path.join(gt_root, scene, "test/ours_30000/test_preds_1/")
        output_root_dir = os.path.join(gt_root, scene, "pseudo_gt")
        for scale in [8, 4, 2]:
            print("scale:", scale)
            output_dir = os.path.join(output_root_dir, f"resize_x{scale}", "images")
            os.makedirs(output_dir, exist_ok=True)
            # copy colmap
            command = f"cp -r {source}/{scene}/sparse/ {output_root_dir}/resize_x{scale}/"
            result = subprocess.run(command, shell=True)
            # target (H, W)
            gt_input_dir = os.path.join(gt_root, scene, f"test/ours_30000/gt_{scale}/")
            gt_files = os.listdir(gt_input_dir)
            first_image = cv2.imread(os.path.join(gt_input_dir, gt_files[0]))
            H, W, _ = first_image.shape
            # resize
            for filename in os.listdir(input_dir):
                if filename.endswith(".png"):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    image = cv2.imread(input_path)
                    resized_image = cv2.resize(image, (W, H))
                    cv2.imwrite(output_path, resized_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument('--source', type=str, default="../3D-Gaussian/360_v2", help="./sparse")
    parser.add_argument('--gt_root', type=str, default="benchmark_360v2_stmt_down",help="imgs")

    args = parser.parse_args()
    scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room", "flowers", "treehill"]
    main(args.source, args.gt_root, scenes)

