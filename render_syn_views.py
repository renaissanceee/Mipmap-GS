#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from random import randint
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json

def write_syn_views(cams, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.append(cams)
    with open(file_path, 'w') as file:
        json.dump(cams, file, indent=4)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scale_factor, scale):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, scale: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                   background, scale_factor=scale_factor, scale=scale)
        # ------------ syn_views ------------
        syn_views = 50 # add 50 more views from track
        render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), f"test_preds_{scale}")
        file_path = os.path.join(render_path,'syn_views.json')
        cam_data_json=[]
        for idx in range(syn_views):
            pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            cam_data = {
                'R': pseudo_cam.R.tolist(),
                'T': pseudo_cam.T.tolist(),
                'FoVx': pseudo_cam.FoVx,
                'FoVy': pseudo_cam.FoVy,
                'image_width': pseudo_cam.image_width,
                'image_height': pseudo_cam.image_height,
                'image_name': '{0:05d}'.format(idx)+ ".png"
            }
            cam_data_json.append(cam_data)
            rendering_syn_views = render(pseudo_cam, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering_syn_views, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        with open(file_path, 'w') as file:
            json.dump(cam_data_json, file, indent=4)
        print("syn_views saved.")
        # ------------------------------------


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scale", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.scale)