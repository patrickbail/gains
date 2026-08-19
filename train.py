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

import os
import torch
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_initial, render_pbr
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth

from glob import glob
from lift_mask_feat import lift_mask_feat
from utils.render_utils import generate_path
from diff_guidance_model import GuidanceModel

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def rotation_geodesic_distance(R1, R2):
    R_diff = R1 @ R2.transpose(0, 1)
    trace_val = torch.clamp((torch.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return torch.acos(trace_val)

def get_closest_cam(viewpoint_stack_persistent, nv_cam):
    min_score = float('inf')
    closest_cam = None
    
    for cam in viewpoint_stack_persistent:
        trans_dist = torch.norm(nv_cam.T - cam.T)
        rot_dist = rotation_geodesic_distance(nv_cam.R, cam.R)
        score = trans_dist + rot_dist
        
        if score < min_score:
            min_score = score
            closest_cam = cam
    
    return closest_cam

def load_diff_model(opt):
    diff_model = GuidanceModel("cuda", loss_type=opt.diff_loss, is_latent=False)
    prompt = ""
    diff_model.embed_prompts(prompt)
    return diff_model

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, args, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    print(args.strength_La, args.r_reduc)
    exit()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 500 #1000

    USE_ENV_SCOPE = opt.use_env_scope
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(dataset.sh_degree, novel_env_root_dir=args.novel_env_root_dir, args=args)
    set_gaussian_para(gaussians, opt) # #
    env_name = None
    if args.envmap_path:
        env_name = os.path.basename(args.envmap_path)
    nv_envs = None
    if args.novel_env_root_dir:
        nv_envs = sorted([env_path for env_path in glob(args.novel_env_root_dir) if "sunset" not in os.path.basename(env_path)])
    scene = Scene(args, gaussians, env_map=env_name)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    lifted_seg = False
    if checkpoint:
        print(f"Loading checkpoint {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.lift_seg:
            seg_feats_path_obj = os.path.join("/".join(checkpoint.split("/")[:-1]), "seg_feats_obj.pt")
            seg_feats_path_part = os.path.join("/".join(checkpoint.split("/")[:-1]), "seg_feats_part.pt")
            seg_feats_path_subpart = os.path.join("/".join(checkpoint.split("/")[:-1]), "seg_feats_subpart.pt")
            if os.path.exists(seg_feats_path_obj) and os.path.exists(seg_feats_path_part) and os.path.exists(seg_feats_path_subpart):
                gaussians._seg_feats_obj = torch.load(seg_feats_path_obj).float().cuda()
                gaussians._seg_feats_part = torch.load(seg_feats_path_part).float().cuda()
                gaussians._seg_feats_subpart = torch.load(seg_feats_path_subpart).float().cuda()
                #gaussians._seg_feats = torch.cat([gaussians._seg_feats_obj, gaussians._seg_feats_part, gaussians._seg_feats_subpart], dim=0)
                if not args.seg_test:
                    lifted_seg = True
                print("Loaded segmentation features")
            else:
                gaussians._seg_feats_obj = torch.zeros((0, 1), dtype=torch.float, device="cuda")
                gaussians._seg_feats_part = torch.zeros((0, 1), dtype=torch.float, device="cuda")
                gaussians._seg_feats_subpart = torch.zeros((0, 1), dtype=torch.float, device="cuda")
                #gaussians._seg_feats= torch.zeros((0, 1), dtype=torch.float, device="cuda")
                print("No segmentation features found")

    if scene.light_rotate:
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    diff_model = None
    if opt.lambda_diff:
        if args.text_cond:
            diff_model = GuidanceModel("cuda", loss_type=opt.diff_loss, is_latent=False)
            prompt = ""
            diff_model.embed_prompts(prompt)
        else:
            diff_model = GuidanceModel("cuda", gd_model_id='lambdalabs/sd-image-variations-diffusers', loss_type=opt.diff_loss, is_latent=False, text_cond=False)
            sparse_viewpoint_stack_persistent = scene.getSparseCameras().copy()
        if diff_model.loss_type == "vsd":
            phi_optimizer = torch.optim.AdamW([{"params": diff_model.phi_params, "lr": 0.0001}], lr=0.0001)
        print(f"Diffusion model loaded and {opt.diff_loss} loss used and text prompt is {bool(args.text_cond)}")

    if args.lambda_diff:
        nv_viewpoint_stack = generate_path(scene.getSparseCameras().copy(), 120)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viz_saved = 0

    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_psnr_for_log = 0.0

    ema_iid_for_log = 0.0
    ema_tv_env_for_log = 0.0
    ema_diff_for_log = 0.0
    ema_intra_seg_for_log = 0.0

    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="")
    if not checkpoint:
        first_iter += 1
    iteration = first_iter
    total_images = len(scene.getTrainCameras())

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')
    print(f'Total images used: {total_images}')
    print(f'sRGB: {opt.srgb}')


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0


    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration, initial_stage, opt.init_until_iter)

        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration == opt.init_until_iter:
            print("Beginning stage two")
            initial_stage = False

            #gaussians.init_shading_normals()

            if not args.joint_geo_mat:
                gaussians._xyz.requires_grad = False
                gaussians._features_dc.requires_grad = False
                gaussians._features_rest.requires_grad = False
                gaussians._opacity.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False

            if (args.lift_seg and not lifted_seg):
                print("Lifting 2D masks...")
                del diff_model
                lift_mask_feat(gaussians, scene, background, pipe, op, args, isObjectScene=args.scene_level == "object", n_frames=args.seg_views)
                lifted_seg = True
                diff_model = load_diff_model(opt)

        # Initialize envmap
        if not initial_stage:
            envmap = gaussians.get_envmap 
            envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # Set render
        render = select_render_method(initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt, render_seg=lifted_seg, scene_level=args.scene_level)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        nv_diff_render_pkg = None
        # iteration % args.diff_freq == 0
        if args.lambda_diff and ((iteration > args.diff_at_iter and iteration > args.init_until_iter) or (args.full_sparse and not args.no_diff_first and args.geo_diff_at_iter < iteration < args.init_until_iter)):
            nv_diff_viewpoint_cam = nv_viewpoint_stack[randint(0, len(nv_viewpoint_stack) - 1)]
            #nv_diff_viewpoint_cam = get_closest_cam(nv_viewpoint_stack, viewpoint_cam)
            if not opt.text_cond and args.lambda_diff:
                embed_cam = get_closest_cam(sparse_viewpoint_stack_persistent, nv_diff_viewpoint_cam)
                diff_model.embed_image(embed_cam.original_image.unsqueeze(0))
            nv_env_id = 0
            if nv_envs and iteration >= args.init_until_iter:
                nv_env_id = -1
            nv_diff_render_pkg = render(nv_diff_viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt, nv_env=nv_env_id, render_seg=lifted_seg, scene_level=args.scene_level)

        gt_image = viewpoint_cam.original_image.cuda()

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration, diff_model, nv_diff_render_pkg, args)
        normal_loss, loss, Ll1 = tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"]
        iid_loss, tv_env_loss, diff_loss, intra_seg_loss = tb_dict["loss_iid"], tb_dict["loss_tv_env"], tb_dict["loss_diff"], tb_dict["loss_intra_seg"]

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()
        if nv_diff_render_pkg:
            if diff_model.loss_type == "vsd":
                phi_optimizer.zero_grad()
                loss_phi = diff_model.loss_phi(nv_diff_render_pkg["render"].unsqueeze(0))
                loss_phi.backward()
                phi_optimizer.step()
        iter_end.record()


        with torch.no_grad():
            
            if viz_saved < args.sparse:
                viz_saved += 1
                save_image(gt_image, os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}.png"))
                if args.lambda_mono_normal:
                    save_image((viewpoint_cam.mono_normal.permute(2, 0, 1)+ 1) * 0.5, os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_normal.png"))
                if args.full_sparse:
                    save_image(viewpoint_cam.mono_depth.unsqueeze(0), os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_depth.png"))
                #save_image(render_pkg["seg_image"], os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_seg.png"))
                if args.lambda_iid:
                    save_image(viewpoint_cam.albedo.cuda(), os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_albedo.png"))
                    if not args.albedo_only_iid:
                        save_image(viewpoint_cam.roughness.cuda(), os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_roughness.png"))
                        if args.include_metallic:
                            save_image(viewpoint_cam.metallic.cuda(), os.path.join(args.visualize_path, f"sparse_{viewpoint_cam.image_name}_metallic.png"))

            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage, lifted_seg=lifted_seg, nv_diff_render_pkg=nv_diff_render_pkg, args=args)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log

            ema_iid_for_log = 0.4 * iid_loss + 0.6 * ema_iid_for_log
            ema_tv_env_for_log = 0.4 * tv_env_loss + 0.6 * ema_tv_env_for_log
            ema_diff_for_log = 0.4 * diff_loss + 0.6 * ema_diff_for_log
            ema_intra_seg_for_log = 0.4 * intra_seg_loss + 0.6 * ema_intra_seg_for_log

            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "ICC": f"{ema_intra_seg_for_log:.{5}f}",
                    "IID": f"{ema_iid_for_log:.{5}f}",
                    "Diff": f"{ema_diff_for_log:.{5}f}",

                    "TV Env": f"{ema_tv_env_for_log:.{5}f}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                if not lifted_seg:
                    gaussians._seg_ids_obj = torch.zeros(gaussians.get_xyz.shape[0], 1)
                    gaussians._seg_ids_part = torch.zeros(gaussians.get_xyz.shape[0], 1)
                    gaussians._seg_ids_subpart = torch.zeros(gaussians.get_xyz.shape[0], 1)
                scene.save(iteration)
            num_gauss = len(gaussians._xyz)

            # Densification
            if iteration < opt.densify_until_iter and num_gauss < 500_000:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                opacity_reset_intval = 1000
                densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    #opt.opacity_cull opt.prune_opacity_threshold 0.005
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                                                size_threshold)

                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity0()

            if total_images < 64 and iteration < opt.init_until_iter and args.lambda_bce > 0 and (args.full_sparse and iteration % 1000 == 0 or (dataset.white_background and iteration == opt.init_until_iter)) and "ref_real" not in args.source_path:
                gaussians.remove_outliers(opt, iteration, linear=True)
                            
            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")
        if lifted_seg and args.seg_test:
            exit()
        iteration += 1








# ============================================================
# Utils for training


def select_render_method(initial_stage):

    if initial_stage:
        render = render_initial
    else:
        render = render_pbr

    return render


def set_gaussian_para(gaussians, opt):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr




def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage, lifted_seg=False, nv_diff_render_pkg=None, args=None):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt, render_seg=lifted_seg, scene_level=args.scene_level)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                #render_pkg["shading_normal"] * 0.5 + 0.5,
                render_pkg["render_image_env"], 
                render_pkg["render_sh"],
                error_map, 
            ]

        if lifted_seg:
            visualization_list.append(render_pkg["seg_image"])
        if nv_diff_render_pkg is not None:
            visualization_list.append(nv_diff_render_pkg['render_image_env'])
  

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            env_dict = gaussians.render_env_map()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,15000,20000,24000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000,24000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.init_until_iter:
        args.save_iterations.append(args.init_until_iter)
        args.checkpoint_iterations.append(args.init_until_iter)
    args.checkpoint_iterations.append(args.iterations)

    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    
    if not args.model_path:
        current_time = datetime.now().strftime('%m%d_%H%M')
        last_subdir = os.path.basename(os.path.normpath(args.source_path))
        args.model_path = os.path.join("./output/", f"{last_subdir}/", f"{last_subdir}-{current_time}")

    print("Optimizing " + args.model_path)

    safe_state(args.quiet, seed=args.seed)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path, args)
    print("\nTraining complete.")