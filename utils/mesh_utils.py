#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
from tqdm import tqdm
from utils.render_utils import save_img_u8
from functools import partial
import open3d as o3d

from utils.loss_utils import ssim as get_ssim
from utils.loss_utils import psnr as get_psnr
from utils.image_utils import mse as get_mse
from lpips import LPIPS
from utils.image_utils import visualize_depth
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    #print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    #print("num vertices raw {}".format(len(mesh.vertices)))
    #print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None, opt=None, train=True, align_albedo=False, align_relighting=False, isSrgb=False, render_initial=False):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.opt = opt
        self.train = train
        self.align_albedo = align_albedo
        self.align_relighting = align_relighting
        self.isSrgb = isSrgb
        self.lpips_fn = LPIPS(net="vgg").cuda()
        self.render_seg = (self.gaussians._seg_ids_subpart.shape[-1] > 1 or self.gaussians._seg_ids_part.shape[-1] > 1 or self.gaussians._seg_ids_obj.shape[-1] > 1)
        self.render_initial = render_initial
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.rgbenvs = []
        self.viewpoint_stack = []

        if not self.train:
            self.gt_albedo = []
            self.pred_albedo = []
            self.base_maps = []
            self.diff_maps = []
            self.spec_maps = []
            self.refl_maps = []
            self.roughness_maps = []
            self.alpha_maps = []
            self.normal_maps = []
            self.surf_normal_maps = []
            self.seg_images = []

            self.shading_maps = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, test=False, visualize=False, scene_level=None, do_roughness=False, relit=False):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        self.albedo_scales = {}
        self.visualize = visualize

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            if self.align_albedo and test and not relit:
                mask = viewpoint_cam.gt_alpha_mask.cpu()
                gt_albedo = viewpoint_cam.albedo[0:3, :, :]
                if self.isSrgb:
                    gt_albedo = srgb_to_rgb(gt_albedo)
                self.gt_albedo.append(gt_albedo.permute(1, 2, 0)[mask > 0].cpu())
            if self.opt:
                render_pkg = self.render(viewpoint_cam, self.gaussians, srgb=self.opt.srgb, opt=self.opt, render_seg=True, scene_level=scene_level)
            else:
                render_pkg = self.render(viewpoint_cam, self.gaussians, render_seg=True, scene_level=scene_level)

            if not self.visualize:
                rgb = render_pkg['render']
                if not self.render_initial and not relit:
                    base_color = render_pkg["base_color_map"]
                    if do_roughness:
                        roughness = render_pkg["roughness_map"].repeat(3, 1, 1)
                    normal = render_pkg["rend_normal"] * 0.5 + 0.5 
                else:
                    depth = render_pkg["surf_depth"]

                self.rgbmaps.append(rgb)
                if not self.render_initial and not relit:
                    self.base_maps.append(base_color) # Albedo
                    if do_roughness:
                        self.roughness_maps.append(roughness.cpu())
                    if self.align_albedo and test:
                        render_albedo = base_color.cpu()
                        if self.isSrgb:
                            render_albedo = srgb_to_rgb(render_albedo)
                        self.pred_albedo.append(render_albedo.permute(1, 2, 0)[mask > 0].cpu())
                    self.normal_maps.append(normal.cpu())
                else:
                    self.depthmaps.append(depth.cpu())
                diff = render_pkg["diffuse_map"]
                spec = render_pkg["specular_map"]
                shading_map = render_pkg["shading_map"]
                self.diff_maps.append(diff.cpu())
                self.spec_maps.append(spec.cpu())
                self.shading_maps.append(shading_map.cpu())
            else:
                rgb = render_pkg['render']
                rgb_env = render_pkg['render_image_env']
                base_color = render_pkg["base_color_map"]
                diff = render_pkg["diffuse_map"]
                spec = render_pkg["specular_map"]
                refl = render_pkg["refl_strength_map"].repeat(3, 1, 1) 
                roughness = render_pkg["roughness_map"].repeat(3, 1, 1)
                alpha = render_pkg["rend_alpha"].repeat(3, 1, 1)
                depth = visualize_depth(render_pkg["surf_depth"])  
                normal = render_pkg["rend_normal"] * 0.5 + 0.5  
                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5 

                self.rgbmaps.append(rgb.cpu())
                self.rgbenvs.append(rgb_env.cpu())
                self.base_maps.append(base_color.cpu()) # Albedo
                if self.render_seg:
                    seg_image = render_pkg['seg_image']
                    self.seg_images.append(seg_image.cpu())
                if self.align_albedo and test:
                    render_albedo = base_color.cpu()
                    if self.isSrgb:
                        render_albedo = srgb_to_rgb(render_albedo)
                    self.pred_albedo.append(render_albedo.permute(1, 2, 0)[mask > 0])
                if self.opt.indirect:
                    indirect_color = render_pkg['indirect_color']
                    self.indirect_color.append(indirect_color.cpu())
                self.diff_maps.append(diff.cpu())
                self.spec_maps.append(spec.cpu())
                self.refl_maps.append(refl.cpu())
                self.roughness_maps.append(roughness.cpu())
                self.alpha_maps.append(alpha.cpu())
                self.depthmaps.append(depth.cpu())
                self.normal_maps.append(normal.cpu())
                self.surf_normal_maps.append(surf_normal.cpu())
        
        if self.align_albedo and test and not relit:
            albedo_gts = torch.cat(self.gt_albedo, dim=0).cpu()
            albedo_ours = torch.cat(self.pred_albedo, dim=0).cpu()

            ratios = (albedo_gts.clamp_min(1e-6) / albedo_ours.clamp_min(1e-6))
            """
            params = []
            for c in range(3):
                g = albedo_gts[:, c]
                p = albedo_ours[:, c]

                mu_g = g.mean()
                mu_p = p.mean()
                var_p = p.var(unbiased=False)

                if var_p < 1e-8:
                    s = torch.tensor(0.0).cuda()
                else:
                    cov = ((p - mu_p) * (g - mu_g)).mean()
                    s = cov / var_p
                params.append(s)

            params = torch.tensor(params)
            params = params / params.mean()
            eps = 1e-8
            numerator = (albedo_ours * albedo_gts).sum(dim=0)
            denominator = albedo_ours.square().sum(dim=0).clamp_min(eps)
            scale = numerator / denominator
            scales = []
            for gt, pred in zip(self.gt_albedo, self.pred_albedo):
                # gt/pred are already masked foreground pixels
                num = (pred * gt).sum(dim=0)
                den = pred.square().sum(dim=0).clamp_min(1e-8)
                scales.append(num / den)
            scale_6 = torch.stack(scales).mean(dim=0)
            self.albedo_scales["6"] = scale_6.tolist()
            """

            self.albedo_scales["0"] = [1.0, 1.0, 1.0]
            self.albedo_scales["1"] = [(ratios)[..., 0].median().item()] * 3
            self.albedo_scales["2"] = (ratios).median(dim=0).values.tolist()
            self.albedo_scales["3"] = (ratios).mean(dim=0).tolist()
            #self.albedo_scales["4"] = params.tolist()
            #self.albedo_scales["5"] = scale.tolist()

        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        #print(f"The estimated bounding radius is {self.radius:.2f}")
        #print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        #print("Running tsdf volume integration ...")
        #print(f'voxel_size: {voxel_size}')
        #print(f'sdf_trunc: {sdf_trunc}')
        #print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        #for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
        for i, cam_o3d in enumerate(to_cam_open3d(self.viewpoint_stack)):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask.unsqueeze(0) < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path, isTest=False, source_path=None, bg_color=None):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            if isTest:
                gt = viewpoint_cam.original_image[0:3, :, :]
                save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                if "tensorIR" in source_path or "Synthetic4Relight" in source_path:
                    gt_albedo = viewpoint_cam.albedo[0:3, :, :]
                    save_img_u8(gt_albedo.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + "_albedo.png"))
                    save_img_u8(srgb_to_rgb(gt_albedo).permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + "_albedo_rgb.png"))
                if "Synthetic4Relight" in source_path:
                    gt_roughness = viewpoint_cam.roughness
                    save_img_u8(gt_roughness.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + "_roughness.png"))

            if bg_color is None:
                save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'render_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.rgbenvs[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'render_env_{0:05d}'.format(idx) + ".png"))
                if self.align_albedo:
                    if self.isSrgb:
                        pred_albedo = rgb_to_srgb(self.apply_linear_alignment(srgb_to_rgb(self.base_maps[idx]), self.albedo_scales["2"], viewpoint_cam.gt_alpha_mask))
                    else:
                        pred_albedo = self.apply_linear_alignment(self.base_maps[idx], self.albedo_scales["2"], viewpoint_cam.gt_alpha_mask)
                    save_img_u8(pred_albedo.permute(1,2,0).cpu().numpy(), 
                                os.path.join(render_path, 'base_{0:05d}'.format(idx) + ".png"))
                else:
                    save_img_u8(self.base_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'base_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.refl_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'refl_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.roughness_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'roughness_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.normal_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'normal_{0:05d}'.format(idx) + ".png"))
            else:
                alpha_map = self.alpha_maps[idx].cuda()
                save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'render_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.rgbenvs[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'render_env_{0:05d}'.format(idx) + ".png"))
                if self.align_albedo:
                    alb = (self.base_maps[idx].cuda() + bg_color[:, None, None].cuda() * (1 - alpha_map))
                    if self.isSrgb:
                        pred_albedo = rgb_to_srgb(self.apply_linear_alignment(srgb_to_rgb(alb), self.albedo_scales["2"], viewpoint_cam.gt_alpha_mask))
                    else:
                        pred_albedo = self.apply_linear_alignment(alb, self.albedo_scales["2"], viewpoint_cam.gt_alpha_mask)
                    save_img_u8(pred_albedo.permute(1,2,0).cpu().numpy(), 
                                os.path.join(render_path, 'base_{0:05d}'.format(idx) + ".png"))
                else:
                    save_img_u8((self.base_maps[idx].cuda() + bg_color[:, None, None].cuda() * (1 - alpha_map)).permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'base_{0:05d}'.format(idx) + ".png"))
                
                save_img_u8(self.refl_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'refl_{0:05d}'.format(idx) + ".png"))
                save_img_u8((self.roughness_maps[idx].cuda() + bg_color[:, None, None].cuda() * (1 - alpha_map)).permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'roughness_{0:05d}'.format(idx) + ".png"))
                save_img_u8((self.normal_maps[idx].cuda() + bg_color[:, None, None].cuda() * (1 - alpha_map)).permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'normal_{0:05d}'.format(idx) + ".png"))
            if self.visualize:
                save_img_u8(self.depthmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'depth_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.diff_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'diff_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.spec_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'spec_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.alpha_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'alpha_{0:05d}'.format(idx) + ".png"))
                save_img_u8(self.surf_normal_maps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'surfnormal_{0:05d}'.format(idx) + ".png"))
                if self.render_seg:
                    save_img_u8(self.seg_images[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, 'seg_{0:05d}'.format(idx) + ".png"))

    def apply_linear_alignment(self, pred, params, mask):
        out = pred.clone()
        mask3 = mask[None, :, :].bool()

        for c in range(3):
            s = params[c]
            # Apply only to masked (foreground) pixels
            out[c][mask3[0]] = torch.clamp(s * pred[c][mask3[0]], 0.0, 1.0)

        return out
    
    def crop_to_mask(self, img, mask):
        # Find valid rows and columns
        rows = torch.any(mask, dim=1)  # shape (H,)
        cols = torch.any(mask, dim=0)  # shape (W,)

        if not rows.any() or not cols.any():
            # No valid pixels, return original
            return img, mask

        row_min, row_max = torch.where(rows)[0][[0, -1]]
        col_min, col_max = torch.where(cols)[0][[0, -1]]

        # Crop img and mask
        cropped_img  = img[:, row_min:row_max+1, col_min:col_max+1]
        cropped_mask = mask[row_min:row_max+1, col_min:col_max+1]

        return cropped_img, cropped_mask

    @torch.no_grad()
    def eval(self, path, source_path=None, score_tracker=None, relit_images=None, use_mask=False, env_name="", scale_id="2", args=None):
        if "tensorIR" in source_path or "Synthetic4Relight" in source_path or "ref_syn" in source_path:
            psnr_avg_alb = 0.0
            ssim_avg_alb = 0.0
            lpips_avg_alb = 0.0
        if "Synthetic4Relight" in source_path:
            mse_avg_rough = 0.0
        if "tensorIR" in source_path or "ref_syn" in source_path:
            mae_avg = 0.0
        psnr_avg = 0.0
        ssim_avg = 0.0
        lpips_avg = 0.0
        relit_scales = None

        if self.align_relighting and relit_images is not None:
            gt_relit_images = []
            pred_relit_images = []
            relit_scales = {}
            for idx, relit_image in tqdm(enumerate(relit_images), desc="calculating aliging score"):
                mask = self.viewpoint_stack[idx].gt_alpha_mask.cpu()
                gt_relit_image = relit_image[0:3, :, :]
                pred_relit_image = self.rgbmaps[idx][0:3, :, :]
                if self.isSrgb:
                    gt_relit_image = srgb_to_rgb(gt_relit_image)
                    pred_relit_image = srgb_to_rgb(pred_relit_image)
                gt_relit_images.append(gt_relit_image.permute(1, 2, 0)[mask > 0])
                pred_relit_images.append(pred_relit_image.permute(1, 2, 0)[mask > 0])

            gt_relit_images = torch.cat(gt_relit_images, dim=0).cpu()#.cuda()
            pred_relit_images = torch.cat(pred_relit_images, dim=0).cpu()#.cuda()

            ratios = (gt_relit_images / pred_relit_images.clamp_min(1e-6))

            """
            params = []
            for c in range(3):
                g = gt_relit_images[:, c]
                p = pred_relit_images[:, c]

                mu_g = g.mean()
                mu_p = p.mean()
                var_p = p.var(unbiased=False)

                if var_p < 1e-8:
                    s = torch.tensor(0.0).cuda()
                else:
                    cov = ((p - mu_p) * (g - mu_g)).mean()
                    s = cov / var_p

                params.append(s)

            eps = 1e-8
            numerator = (pred_relit_images * gt_relit_images).sum(dim=0)
            denominator = pred_relit_images.square().sum(dim=0).clamp_min(eps)
            scale = numerator / denominator

            scales = []
            for gt, pred in zip(gt_relit_images, pred_relit_images):
                # gt/pred are already masked foreground pixels
                num = (pred * gt).sum(dim=0)
                den = pred.square().sum(dim=0).clamp_min(1e-8)
                scales.append(num / den)
            scale_6 = torch.stack(scales).mean(dim=0)
            self.albedo_scales["6"] = scale_6.tolist()
            """

            relit_scales["0"] = [1.0, 1.0, 1.0]
            relit_scales["1"] = [(ratios)[..., 0].median().item()] * 3
            relit_scales["2"] = (ratios).median(dim=0).values.tolist()
            relit_scales["3"] = (ratios).mean(dim=0).tolist()
            #relit_scales["4"] = params
            #relit_scales["5"] = scale.tolist()

        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="evaluating images"):
            # RGB predictions
            if relit_images is not None:
                image = relit_images[idx]
                gt = image[0:3, :, :].cuda()
            else:
                gt = viewpoint_cam.original_image[0:3, :, :]
            pred = self.rgbmaps[idx].cuda()
            mask = viewpoint_cam.gt_alpha_mask

            if use_mask:
                gt, _  = self.crop_to_mask(gt, mask)

                diffuse, _ = self.crop_to_mask(diffuse, mask)
                specular, _ = self.crop_to_mask(specular, mask)
                shading_map, _ = self.crop_to_mask(shading_map, mask)

                pred, mask = self.crop_to_mask(pred, mask)

            if relit_scales is not None:
                if self.isSrgb:
                    pred = rgb_to_srgb(self.apply_linear_alignment(srgb_to_rgb(pred), relit_scales[scale_id], mask))
                else:
                    pred = self.apply_linear_alignment(pred, relit_scales[scale_id], mask)

            lpips_avg += self.lpips_fn(gt, pred).mean().double()
            ssim_avg += get_ssim(gt, pred).mean().double()
            psnr_avg += get_psnr(gt, pred).mean().double()

            if ("tensorIR" in source_path or "Synthetic4Relight" in source_path or "ref_syn" in source_path) and relit_images is None:
                # Albedo
                mask = viewpoint_cam.gt_alpha_mask
                gt_albedo = viewpoint_cam.albedo[0:3, :, :].cuda()
                pred_albedo = self.base_maps[idx].cuda()

                if use_mask:
                    gt_albedo, _  = self.crop_to_mask(gt_albedo, mask)
                    pred_albedo, mask = self.crop_to_mask(pred_albedo, mask)

                if self.align_albedo:
                    if self.isSrgb:
                        pred_albedo = rgb_to_srgb(self.apply_linear_alignment(srgb_to_rgb(pred_albedo), self.albedo_scales[scale_id], mask))
                    else:
                        pred_albedo = self.apply_linear_alignment(pred_albedo, self.albedo_scales[scale_id], mask)

                lpips_avg_alb += self.lpips_fn(gt_albedo, pred_albedo).mean().double()
                ssim_avg_alb += get_ssim(gt_albedo, pred_albedo).mean().double()
                psnr_avg_alb += get_psnr(gt_albedo, pred_albedo).mean().double()

            # Roughness
            if "Synthetic4Relight" in source_path and relit_images is None:
                gt_roughness = viewpoint_cam.roughness.cuda()
                pred_roughness = self.roughness_maps[idx].cuda()

                if use_mask:
                    mask = viewpoint_cam.gt_alpha_mask 
                    gt_roughness   = gt_roughness.permute(1, 2, 0)[mask > 0]
                    pred_roughness = pred_roughness.permute(1, 2, 0)[mask > 0]

                mse_avg_rough += get_mse(gt_roughness, pred_roughness, masked=use_mask).mean().double()

            # Normals
            if ("tensorIR" in source_path or "ref_syn" in source_path) and relit_images is None:
                gt_normal = viewpoint_cam.mono_normal.cuda()
                pred_normal = self.normal_maps[idx].cuda() * 2.0 - 1.0

                pred_normal = pred_normal / (pred_normal.norm(dim=0, keepdim=True) + 1e-8)
                gt_normal   = gt_normal   / (gt_normal.norm(dim=0, keepdim=True) + 1e-8)

                cos = (pred_normal * gt_normal).sum(dim=0).clamp(-1, 1)
                ang = torch.acos(cos) * 180.0 / torch.pi

                mask = viewpoint_cam.gt_alpha_mask
                mae_avg += ang[mask].mean()
        
        psnr = psnr_avg / len(self.viewpoint_stack)
        ssim = ssim_avg / len(self.viewpoint_stack)
        lpips = lpips_avg / len(self.viewpoint_stack)
        print(f"psnr_avg: {psnr}; ssim_avg: {ssim}; lpips_avg: {lpips}")

        if ("tensorIR" in source_path or "Synthetic4Relight" in source_path or "ref_syn" in source_path) and relit_images is None:
            psnr_alb = psnr_avg_alb / len(self.viewpoint_stack)
            ssim_alb = ssim_avg_alb / len(self.viewpoint_stack)
            lpips_alb = lpips_avg_alb / len(self.viewpoint_stack)
            print(f"psnr_avg_alb: {psnr_alb}; ssim_avg_alb: {ssim_alb}; lpips_avg_alb: {lpips_alb}")
        if "Synthetic4Relight" in source_path and relit_images is None:
            mse_rough = mse_avg_rough / len(self.viewpoint_stack)
            print(f"mse_rough: {mse_rough}")
        if ("tensorIR" in source_path or "ref_syn" in source_path) and relit_images is None:
            mae = mae_avg / len(self.viewpoint_stack)
            print(f"mae: {mae}")
        if score_tracker is not None:
            score_tracker['psnr'] = psnr
            score_tracker['ssim'] = ssim
            score_tracker['lpips'] = lpips
            return score_tracker
        with open(path, "w+") as eval_file:
            eval_file.write(f"psnr_avg: {psnr}; ssim_avg: {ssim}; lpips_avg: {lpips} \n")
            if "tensorIR" in source_path or "Synthetic4Relight" in source_path or "ref_syn" in source_path:
                eval_file.write(f"psnr_avg_alb: {psnr_alb}; ssim_avg_alb: {ssim_alb}; lpips_avg_alb: {lpips_alb} \n")
            if "Synthetic4Relight" in source_path:
                eval_file.write(f"mse_rough: {mse_rough}")
            if "tensorIR" in source_path or "ref_syn" in source_path:
                eval_file.write(f"mae_normal: {mae}")