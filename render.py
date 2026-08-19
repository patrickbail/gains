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
import os
import numpy as np
import math
from scene import Scene
from gaussian_renderer import render_pbr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos
from scene.light import EnvLight

from torchvision.utils import save_image

from glob import glob
from PIL import Image
from utils.general_utils import PILtoTorch

import warnings
warnings.filterwarnings("ignore")

def load_relit_images(test_cams, envmap_name, args):
    relit_images = []
    for test_cam in test_cams:
        if "tensorIR" in args.source_path or "ref_syn" in args.source_path:
            relit_img_path = test_cam.image_path[:-4] + f"_{envmap_name}.png"
        else:
            img_id = os.path.basename(test_cam.image_path).split("_")[0]
            relit_img_path = os.path.join(os.path.dirname(os.path.dirname(test_cam.image_path)), "test_rli", f"{envmap_name}_{img_id}.png")
        with Image.open(relit_img_path) as relit_image:
            relit_im_data = np.array(relit_image.convert("RGBA"))

        bg = np.array([1,1,1]) if args.white_background else np.array([0, 0, 0])
        relit_norm_data = relit_im_data / 255.0
        relit_arr = relit_norm_data[:,:,:3] * relit_norm_data[:, :, 3:4] + bg * (1 - relit_norm_data[:, :, 3:4])

        orig_w, orig_h = relit_image.size
        if args.resolution in [1, 2, 4, 8]:
            resolution = math.floor(orig_w/(args.resolution) + 0.5), math.floor(orig_h/(args.resolution) + 0.5)
            scale = float(args.resolution)
        else:  # should be a type that converts to float
            if args.resolution == -1:
                global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        relit_img = Image.fromarray(np.array(relit_arr*255.0, dtype=np.byte), "RGB")
        if len(relit_img.split()) > 3:
            resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in relit_img.split()[:3]], dim=0)
            relit_image = resized_image_rgb.cuda()
        else:
            resized_image_rgb = PILtoTorch(relit_img, resolution)
            relit_image = resized_image_rgb.cuda()

        relit_images.append(relit_image.float())
    return torch.stack(relit_images, dim=0)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--align_albedo", action="store_true")
    parser.add_argument("--align_relighting", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--scale_id", default="2", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args=args)
    env_name = None
    if args.envmap_path:
        env_name = args.envmap_path.split("/")[-1][:-4]
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, env_map=env_name, skip_train=args.skip_train if not args.full_sparse else False, test=True)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.relight and args.envmap_path:
        gaussians.env_map = EnvLight(path=args.envmap_path, device='cuda', max_res=1024).cuda()
        gaussians.env_map.build_mips()
    if scene.isBlender:
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
    elif args.relight:
        transform = torch.tensor([
            [1, 0, 0], 
            [0, -1, 0], 
            [0, 0, 1]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    root_dir_ext = f"_score{args.scale_id}"
    if env_name:
        root_dir_ext = root_dir_ext + f"_{env_name}"
    if args.srgb:
        root_dir_ext = root_dir_ext + "_srgb"
    if args.use_mask:
        root_dir_ext = root_dir_ext + "_masked"
    if args.align_albedo:
        root_dir_ext = root_dir_ext + "_alAlb"
    test_dir = os.path.join(args.model_path, "test" + root_dir_ext, "ours_{}".format(scene.loaded_iter))
    render = render_pbr

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, opt=args, train=False, 
                                       align_albedo=args.align_albedo, align_relighting=args.align_relighting, isSrgb=args.srgb)    
    
    if not args.skip_train:
        print("training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        if args.export:
            gaussExtractor.export_image(train_dir, source_path=args.source_path, bg_color=background.cpu())
        else:
            gaussExtractor.eval(os.path.join(train_dir, "eval.txt"), args.source_path, use_mask=args.use_mask, scale_id=args.scale_id)
        
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        test_tray = scene.getTestCameras()
        gaussExtractor.reconstruction(test_tray, visualize=args.export, test=True, do_roughness="Synthetic4Relight" in args.source_path)
        if args.export:
            gaussExtractor.export_image(test_dir, isTest=True, source_path=args.source_path, bg_color=background.cpu())
        else:
            gaussExtractor.eval(os.path.join(test_dir, "eval.txt"), args.source_path, use_mask=args.use_mask, scale_id=args.scale_id, args=args)
    
    if args.render_path:
        print("render videos ...")
        if env_name:
            traj_dir = os.path.join(args.model_path, f'traj_{env_name}', "ours_{}".format(scene.loaded_iter))
        else:
            traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)

        env_dict = gaussians.render_env_map()
        save_image(env_dict["env1"].detach().permute(2, 0, 1).detach(), os.path.join(traj_dir, "env.png"))

        n_fames = 240
        cams = [cam for i, cam in enumerate(scene.getTrainCameras())]
        cam_traj = generate_path(cams, n_frames=n_fames)

        n_fames = len(cam_traj)
        gaussExtractor.reconstruction(cam_traj, visualize=True, scene_level=args.scene_level)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames, render_seg=gaussExtractor.render_seg, indirect_color=op.indirect)

    if args.test_relight and (len(scene.getTestCameras()) > 0):
        print("testing under multiple relight conditions...")
        if "tensorIR" in args.source_path:
            light_name_list = ["city", "fireplace", "night", "forest"]
            envmap_paths = glob(os.path.join(args.test_relight, "*.hdr"))
        elif "ref_syn" in args.source_path:
            light_name_list = ["bridge", "city", "forest"]
            envmap_paths = glob(os.path.join(args.test_relight, "*.hdr"))
        else:
            light_name_list = ["envmap6", "envmap12"]
            envmap_paths = glob(os.path.join(args.test_relight, "*.exr"))
        envmap_paths = [path for path in envmap_paths if os.path.basename(path).split(".")[0] in light_name_list]

        root_dir_ext = f"_score{args.scale_id}"
        if args.srgb:
            root_dir_ext = root_dir_ext + "_srgb"
        if args.use_mask:
            root_dir_ext = root_dir_ext + "_masked"
        if args.align_relighting:
            root_dir_ext = root_dir_ext + "_alRelit"

        test_relight_dir = os.path.join(args.model_path, 'test_relight' + root_dir_ext, "ours_{}".format(scene.loaded_iter))
        os.makedirs(test_relight_dir, exist_ok=True)

        scores = {}
        for envmap_path in envmap_paths:
            envmap_name = os.path.basename(envmap_path).split(".")[0]
            print(envmap_name)
            gaussExtractor.gaussians.set_envmap(envmap_path)
            test_cams = scene.getTestCameras()
            gaussExtractor.reconstruction(test_cams, test=True, relit=True)

            relit_images = load_relit_images(test_cams, envmap_name, args).cpu()
            results = gaussExtractor.eval(os.path.join(test_dir, "eval.txt"), args.source_path, score_tracker={}, relit_images=relit_images, use_mask=args.use_mask, env_name=envmap_name, scale_id=args.scale_id, args=args)
            print(f"Envmap {envmap_name}: {results}")
            scores[envmap_name] = results
            del relit_images
            torch.cuda.empty_cache()

        num_envmaps = len(envmap_paths)
        avg_psnr = sum(m['psnr'].item() for m in scores.values()) / num_envmaps
        avg_ssim = sum(m['ssim'].item() for m in scores.values()) / num_envmaps
        avg_lpips = sum(m['lpips'].item() for m in scores.values()) / num_envmaps

        with open(os.path.join(test_relight_dir, "eval.txt"), "w+") as eval_file:
            print(f"psnr_avg: {avg_psnr}; ssim_avg: {avg_ssim}; lpips_avg: {avg_lpips}")
            eval_file.write(f"psnr_avg: {avg_psnr}; ssim_avg: {avg_ssim}; lpips_avg: {avg_lpips}")