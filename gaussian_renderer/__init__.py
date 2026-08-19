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
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization_blend_seg import GaussianRasterizationSettings as GaussianRasterizationSettings_custom
from diff_surfel_rasterization_blend_seg import GaussianRasterizer as GaussianRasterizer_custom
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.refl_utils import  get_specular_color_surfel
from utils.graphics_utils import linear_to_srgb
from matplotlib import cm

def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal
    }

def render_initial(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, nv_env=None, render_seg=False, scene_level=None, nv_cam=None, sh_degree=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_depth_median = regularizations['render_depth_median']
    render_depth_expected = regularizations['render_depth_expected']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        rendered_image = linear_to_srgb(rendered_image)
    final_image = rendered_image + bg_color[:, None, None] * (1 - render_alpha)

    pc.use_nv_env = -1
    envmap = pc.get_envmap
    rays_d = viewpoint_camera.rays_d

    if viewpoint_camera.gt_alpha_mask is not None:
        alpha = viewpoint_camera.gt_alpha_mask.float()
    else:
        alpha = render_alpha
    persp_env_light = envmap(rays_d, mode='pure_env', nv_cam=nv_cam).permute(2,0,1)
    render_image_env = final_image * alpha + persp_env_light * (1 - alpha)

    rets =  {"render": final_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'render_depth_expected': render_depth_expected,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'render_image_env': render_image_env
    }

    return rets

def render_pbr(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, nv_env=None, render_seg=False, nv_cam=False, scene_level=None, sh_degree=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    sh_degree = sh_degree if sh_degree is not None else pc.active_sh_degree

    raster_settings = GaussianRasterizationSettings_custom(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_seg=render_seg
    )

    rasterizer = GaussianRasterizer_custom(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    refl = pc.get_refl
    ori_color = pc.get_ori_color
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    normals = pc.get_normal_2(scaling_modifier, dir_pp_normalized)
    
    seg_ids_obj = pc.get_seg_ids.cuda()
    seg_ids_part = pc.get_seg_ids_part.cuda()
    seg_ids_subpart = pc.get_seg_ids_subpart.cuda()
    contrib, rendered_image, rendered_features, radii, allmap, alpha_contrib, seg_blend_obj, seg_blend_part, seg_blend_subpart = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = torch.cat((refl, roughness, ori_color, normals), dim=-1),
        seg_ids=seg_ids_obj,
        seg_ids_part=seg_ids_part,
        seg_ids_subpart=seg_ids_subpart,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    seg_image = torch.zeros_like(rendered_image)
    seg_masks = torch.zeros_like(rendered_image)
    if render_seg:
        if scene_level == "object":
            seg_ids = seg_ids_obj
            seg_blend = seg_blend_obj
        else:
            seg_ids = seg_ids_subpart
            seg_blend = seg_blend_subpart

        # Bool segmentation masks
        max_indices = torch.argmax(seg_blend, dim=0)  # shape: (H, W)
        one_hot = torch.nn.functional.one_hot(max_indices, num_classes=seg_blend.shape[0])  # (H, W, K)
        seg_masks = one_hot.permute(2, 0, 1).bool()#.float() # K, H, W
        # Color visual
        seg_blend = seg_blend.permute(1, 2, 0).argmax(dim=-1)
        cmap = cm.get_cmap("turbo", seg_ids.shape[-1])
        colors = (torch.tensor(cmap(range(seg_ids.shape[-1]))[:, :3]) * 255).cuda() # (K, 3)
        seg_image = colors[seg_blend].permute(2, 0, 1)/255.0

    render_sh = rendered_image
    refl_strength = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    shading_normals = rendered_features[5:8]

    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)

    #shading_normal_map = shading_normals.permute(1,2,0)
    #shading_normal_map = shading_normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)

    # Specular
    pc.use_nv_env = nv_env
    if nv_cam:
        albedo = albedo.detach()
    envmap = pc.get_envmap
    specular, rays_d = get_specular_color_surfel(envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, normal_map, render_alpha.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), viewpoint_cam=viewpoint_camera)

    # Diffuse
    diffuse_light = envmap(normal_map.contiguous(), mode="diffuse", nv_cam=nv_cam).permute(2, 0, 1)
    diffuse = diffuse_light * albedo * (1-refl_strength)

    # Final image
    final_image = diffuse + specular
    
    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        final_image = linear_to_srgb(final_image)
        albedo = linear_to_srgb(albedo)
        specular = linear_to_srgb(specular)
        diffuse = linear_to_srgb(diffuse)
        render_sh = linear_to_srgb(render_sh)


    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)
    render_sh = render_sh + bg_color[:, None, None] * (1 - render_alpha)

    persp_env_light = envmap(rays_d, mode='pure_env', nv_cam=nv_cam).permute(2,0,1)
    render_image_env = final_image * render_alpha + persp_env_light * (1 - render_alpha)

    results =  {"render": final_image,
            "refl_strength_map": refl_strength,
            "diffuse_map": diffuse,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            #'shading_normal': shading_normals,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'alpha_contrib': alpha_contrib.squeeze(0),
            'seg_image': seg_image,
            'seg_masks': seg_masks,
            'render_sh': render_sh,
            'render_image_env': render_image_env,
            'shading_map': diffuse_light
    }
    
    return results
