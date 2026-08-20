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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from .image_utils import psnr
import torchvision.transforms as transforms
from torchmetrics.functional.regression import pearson_corrcoef
from utils.graphics_utils import srgb_to_linear

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_seg_intra_error(render_pkg, seg_classes, region, region_2, args, include_albedo=False, include_error_bias=False, gt_img=None):
    seg_masks = render_pkg["seg_masks"] # [K, H, W]
    seg_masks_2 = render_pkg["seg_masks"].view(seg_classes, -1)
    rendered_roughness = render_pkg["roughness_map"].view(1, -1)
    rendered_metallic  = render_pkg["refl_strength_map"].view(1, -1)

    if include_albedo:
        rendered_albedo = render_pkg["base_color_map"].view(3, -1)

    if include_error_bias:
        error_map = torch.abs(gt_img - render_pkg["render"])

    intra_loss_r = 0.0
    intra_loss_m = 0.0
    intra_loss_a = 0.0
    spec_bias_loss = 0.0
    valid_classes = 0

    region_size = region.sum()
    region_size_2 = region_2.sum()
    region_ratio = region_size / region_size_2

    for k in range(seg_classes):
        mask = seg_masks[k].bool()
        if mask.sum() == 0:
            continue

        overlap_mask = mask & region
        overlap_area = overlap_mask.sum()
        if overlap_area == 0:
            continue

        invalid_mask = mask & (~region)
        invalid_ratio = invalid_mask.sum() / mask.sum()
        if invalid_ratio > 0.5:
            continue

        mask_2 = seg_masks_2[k].bool() & region.view(-1)
        valid_classes += 1

        vals_r = rendered_roughness[0, mask_2]
        vals_m = rendered_metallic[0, mask_2]

        area_ratio = (mask_2.sum() / region_size).clamp(min=1e-4)
        exp_w = 25.0 * region_ratio
        scale = torch.exp(exp_w * area_ratio)

        # Intra-class consistency for Roughness and Metallic
        intra_loss_r += vals_r.var(unbiased=False) * scale
        intra_loss_m += vals_m.var(unbiased=False) * scale

        # Albedo uniformity
        if include_albedo:
            mean_r = vals_r.mean()
            mean_m = vals_m.mean()
            w_spec = (1.0 - mean_r).clamp(0, 1) * mean_m.clamp(0, 1)

            vals = rendered_albedo[:, mask_2]
            mu = vals.mean(dim=1, keepdim=True)
            diff = vals - mu
            sq = (diff * diff).sum(dim=0)
            intra_loss_a += sq.mean() * scale * w_spec

        # Specularity bias using class-wise error
        if include_error_bias:
            class_error = error_map[:, mask].mean()  # average error for this class
            # weight magnitude can be tuned
            spec_bias = class_error * (
                (1.0 - vals_m.mean()) + (args.strength_r * vals_r.mean())
            )
            spec_bias_loss += spec_bias * scale

    intra_class_consistency_loss = (
        (intra_loss_r + intra_loss_m + args.strength_La*intra_loss_a + 0.1*spec_bias_loss) / valid_classes
    )

    return intra_class_consistency_loss

# From FatesGS: https://github.com/yulunwu0108/FatesGS/blob/master/utils/loss_utils.py
def TVLoss(network_output, pred_output, edge_margin=1e-2, margin=1e-4):
    """Total variation loss for a 2D image. input is expected to be of shape (channel, h, w)"""
    h_diff = torch.max((network_output[:, 1:, :] - network_output[:, :-1, :]).abs() - margin,
                       torch.zeros_like(network_output[:, 1:, :])) * ((pred_output[:, 1:, :] - pred_output[:, :-1, :]).abs() < edge_margin).float()
    w_diff = torch.max((network_output[:, :, 1:] - network_output[:, :, :-1]).abs() - margin,
                       torch.zeros_like(network_output[:, :, 1:])) * ((pred_output[:, :, 1:] - pred_output[:, :, :-1]).abs() < edge_margin).float()
    return torch.mean(h_diff) + torch.mean(w_diff)

def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]

    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()

    return patch_rank_loss

# From FatesGS: https://github.com/yulunwu0108/FatesGS/blob/master/utils/loss_utils.py
transform1 = transforms.CenterCrop((576, 768))
transform2 = transforms.CenterCrop((544, 736))
def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        surf_depth_crop = transform(surf_depth)
        mono_depth_crop = transform(mono_depth.unsqueeze(0))

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss

def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    return 1 - torch.nan_to_num(co)

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration, diff_model, nv_diff_render_pkg, args):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()
    gt_alpha_mask = None

    if viewpoint_camera.gt_alpha_mask is not None:
        gt_alpha_mask = viewpoint_camera.gt_alpha_mask
        if args.lambda_bce > 0:
            rendered_normal = rendered_normal * gt_alpha_mask

    # RGB loss
    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    # Normal consistency loss (alpha blended normals vs normals from depth)
    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:# and iteration <= opt.init_until_iter:
        surf_normal = render_pkg['surf_normal']
        if gt_alpha_mask is not None and args.lambda_bce > 0:
            surf_normal = surf_normal * gt_alpha_mask
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    # Segmentation Intra class consistency loss
    if iteration > opt.init_until_iter and opt.lambda_intra_seg:
        seg_classes = render_pkg["seg_masks"].shape[0]

        intra_class_consistency_loss = 0.0
        tb_dict["loss_intra_seg"] = 0.0

        region_2 = torch.ones_like(rendered_depth).squeeze(0).bool()
        include_alb_and_error = True
        if gt_alpha_mask is not None:
            region = gt_alpha_mask > 0.5
        else:
            region = torch.ones_like(rendered_depth).squeeze(0).bool()
        if opt.lambda_intra_seg:
            intra_class_consistency_loss = get_seg_intra_error(render_pkg, seg_classes, region, region_2, args, include_albedo=include_alb_and_error, 
                                                               include_error_bias=include_alb_and_error, gt_img=viewpoint_camera.original_image.cuda())
            tb_dict["loss_intra_seg"] = intra_class_consistency_loss.item()
        loss = loss + opt.lambda_intra_seg * intra_class_consistency_loss
    else:
        tb_dict["loss_intra_seg"] = torch.zeros_like(loss)
        tb_dict["loss_intra_segv2"] = torch.zeros_like(loss)

    # Diffusion loss
    if nv_diff_render_pkg and ((iteration > opt.init_until_iter) or (iteration <= opt.init_until_iter and not args.no_diff_first)):
        step_ratio = None
        if args.step_ratio:
            step_ratio = args.step_ratio
        if "ref_real" not in args.source_path:
            nv_diff_image = nv_diff_render_pkg['render_image_env']
        else:
            nv_diff_image = nv_diff_render_pkg['render']
        if iteration > args.init_until_iter and args.srgb:
            nv_diff_image = srgb_to_linear(nv_diff_image)
        render_path = os.path.join(args.model_path, "visualize")
        loss_diff = diff_model.loss(nv_diff_image.unsqueeze(0), current_iter=iteration, max_iter=opt.iterations, render_dir=render_path, step_ratio=step_ratio, guidance_scale=100)
        tb_dict["loss_diff"] = loss_diff.item()
        loss = loss + opt.lambda_diff * loss_diff
    else:
        tb_dict["loss_diff"] = torch.zeros_like(loss)

    # IID prior loss
    if iteration > opt.init_until_iter and opt.lambda_iid:
        w = max(0.0, 1.0 - (iteration - opt.init_until_iter) / (args.iterations - opt.init_until_iter))

        pred_albedo = viewpoint_camera.albedo.cuda().detach()
        rendered_albedo = render_pkg["base_color_map"]
        
        albedo_color_loss = l2_loss(rendered_albedo, pred_albedo)
        tb_dict["loss_albedo"] = albedo_color_loss.item()

        if not args.albedo_only_iid:
            # Roughness
            pred_roughness = viewpoint_camera.roughness.cuda()
            rendered_roughness = render_pkg["roughness_map"]
            Ll1_roughness_loss = l1_loss(rendered_roughness, pred_roughness)
            tb_dict["loss_roughness"] = Ll1_roughness_loss.item()

            # Metallic
            if args.include_metallic:
                pred_metallic = viewpoint_camera.metallic.cuda()
                rendered_metallic = render_pkg["refl_strength_map"]
                Ll1_metallic_loss = l1_loss(rendered_metallic, pred_metallic)
                tb_dict["loss_metallic"] = Ll1_metallic_loss.item()
            else:
                Ll1_metallic_loss = 0
                tb_dict["loss_metallic"] = torch.zeros_like(loss)
        else:
            Ll1_roughness_loss = 0
            Ll1_metallic_loss = 0
            tb_dict["loss_roughness"] = torch.zeros_like(loss)
            tb_dict["loss_metallic"] = torch.zeros_like(loss)
        
        # Total IID loss
        iid_loss = (albedo_color_loss + Ll1_roughness_loss + Ll1_metallic_loss)
        tb_dict["loss_iid"] = iid_loss.item()
        loss = loss + w * opt.lambda_iid * iid_loss
    else:
        tb_dict["loss_albedo"] = torch.zeros_like(loss)
        tb_dict["loss_roughness"] = torch.zeros_like(loss)
        tb_dict["loss_iid"] = torch.zeros_like(loss)

    # Total Variation loss for enviroment map
    if opt.lambda_tv_env > 0 and iteration > args.init_until_iter:
        envmap = pc.render_env_map()["env1"]
        tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
        tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
        env_tv_loss = tv_h1 + tv_w1
        tb_dict["loss_tv_env"] = env_tv_loss.item()
        loss = loss + args.lambda_tv_env * env_tv_loss
    else:
        tb_dict["loss_tv_env"] = torch.zeros_like(loss)

    # Sparse geometry Losses
    if args.full_sparse and iteration < args.init_until_iter:
        # Depth guidance
        if not args.no_depth:
            pearson_loss = 0.0
            #lp_loss = 0.0
            if viewpoint_camera.gt_alpha_mask is not None:
                mono_depth = viewpoint_camera.mono_depth.cuda()
                surf_depth = render_pkg["surf_depth"]
                disp_mono = 1 / mono_depth[viewpoint_camera.gt_alpha_mask > 0.5].clamp(1e-6) # [N]
                disp_render = 1 / surf_depth[viewpoint_camera.gt_alpha_mask.unsqueeze(0) > 0.5].clamp(1e-6) # [N]
                pearson_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()
            else:
                mono_depth = viewpoint_camera.mono_depth.cuda().detach()
                pearson_loss = pearson_depth_loss(render_pkg["surf_depth"], mono_depth)

            depth_rank_loss = 0.0
            surf_depth = render_pkg["surf_depth"]
            mask = (surf_depth.view(-1) > 0)
            mono_depth = viewpoint_camera.mono_depth.cuda().detach()

            mono_depth = 1 / viewpoint_camera.mono_depth.cuda().detach().clamp(1e-6) # [N]
            surf_depth = 1 / render_pkg["surf_depth"].clamp(1e-6) # [N]

            if viewpoint_camera.gt_alpha_mask is not None:
                object_mask = viewpoint_camera.gt_alpha_mask > 0.5
                mask = mask & object_mask.view(-1)
                depth_rank_loss = get_depth_ranking_loss(surf_depth, mono_depth, object_mask.unsqueeze(0))
            else:
                depth_rank_loss = get_depth_ranking_loss(surf_depth, mono_depth, None)

            loss = loss + args.lambda_dr * depth_rank_loss + args.lambda_pl * pearson_loss

        # Normal guidance
        nsmooth_loss = 0.0
        if args.lambda_mono_normal:# and iteration > 6000:
            #normal_bg = torch.tensor([0, 0, 0]).view(3, 1, 1).cuda()
            if viewpoint_camera.gt_alpha_mask is not None:
                surf_normal = render_pkg["surf_normal"] * gt_alpha_mask
                rend_normal = render_pkg["rend_normal"] * gt_alpha_mask
                mono_normal = viewpoint_camera.mono_normal.cuda().detach().permute(2, 0, 1) * gt_alpha_mask#[[2, 1, 0], :, :]
            else:
                surf_normal = render_pkg["surf_normal"]
                rend_normal = render_pkg["rend_normal"]
                mono_normal = viewpoint_camera.mono_normal.cuda().detach().permute(2, 0, 1)

            nsmooth_loss = TVLoss(surf_normal, mono_normal)
            l1n_loss = l1_loss(surf_normal, mono_normal)

            loss = loss + args.lambda_mono_normal * (l1n_loss + 0.1*l1_loss(rend_normal, mono_normal) + nsmooth_loss)
            
        #shading_normal = render_pkg['shading_normal']
        #if gt_alpha_mask is not None and args.lambda_bce > 0:
        #    shading_normal = shading_normal * gt_alpha_mask
        #loss_shading_normal = (1 - (shading_normal * surf_normal).sum(dim=0)**0.05)[None]
        #loss = loss + loss_shading_normal.mean()
        # Alpha map loss if gt mask is given
        if viewpoint_camera.gt_alpha_mask is not None and iteration < args.init_until_iter and args.lambda_bce > 0:
            bce_loss = F.binary_cross_entropy(render_pkg["rend_alpha"].squeeze(0), viewpoint_camera.gt_alpha_mask.float())
            loss = loss + args.lambda_bce * bce_loss
    
    return loss, tb_dict
