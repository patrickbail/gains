import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from scene.light import EnvLight
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, init_predefined_omega, z_score_from_percentage
from utils.general_utils import build_scaling_rotation, safe_normalize, flip_align_view
from glob import glob
import pytorch3d.ops as p3dops

def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx), 
        torch.sin(gy)*torch.sin(gx), 
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.refl_activation = torch.sigmoid
        self.inverse_refl_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.asg_param = init_predefined_omega(4, 8)


    def __init__(self, sh_degree : int, novel_env_root_dir = "", args=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._refl_strength = torch.empty(0) 
        self._ori_color = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._seg_ids_obj = torch.empty(0)
        self._seg_ids_part = torch.empty(0)
        self._seg_ids_subpart = torch.empty(0)
        self._seg_feats_obj = torch.empty(0)
        self._seg_feats_part = torch.empty(0)
        self._seg_feats_subpart = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self._normal1 = torch.empty(0)
        self._normal2 = torch.empty(0)

        self.optimizer = None
        self.free_radius = 0    
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_refl_value = 0.01
        self.init_roughness_value = 0.1 #[0,1]
        self.init_ori_color = 0  
        self.enlarge_scale = 1.5
        self.refl_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.novel_envs = []
        self.vertices = None
        self.faces = None
        if novel_env_root_dir:
            print("Utilizing novel environment maps")
            self.use_nv_env = None
            env_paths = glob(os.path.join(novel_env_root_dir, "*.hdr"))
            env_paths = sorted([env_path for env_path in env_paths if "sunset" not in env_path])
            for env_path in env_paths:
                novel_env = EnvLight(path=env_path, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=False).cuda()
                novel_env.build_mips()
                if "ref_real" not in args.source_path:
                    transform = torch.tensor([
                        [0, -1, 0], 
                        [0, 0, 1], 
                        [-1, 0, 0]
                    ], dtype=torch.float32, device="cuda")
                    novel_env.set_transform(transform)
                self.novel_envs.append(novel_env)
        self.scene_level = args.scene_level
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._refl_strength,  
            self._roughness, 
            self._ori_color, 
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,  
            self._normal2,  
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._seg_ids_obj,
            self._seg_ids_part,
            self._seg_ids_subpart,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._refl_strength,  
        self._roughness, 
        self._ori_color, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._normal1,  
        self._normal2,  
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._seg_ids_obj,
        self._seg_ids_part,
        self._seg_ids_subpart) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_refl(self): 
        return self.refl_activation(self._refl_strength)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_ori_color(self): 
        return self.color_activation(self._ori_color)
    
    @property
    def get_seg_ids(self): 
        return self._seg_ids_obj
    
    @property
    def get_seg_ids_part(self): 
        return self._seg_ids_part
    
    @property
    def get_seg_ids_subpart(self): 
        return self._seg_ids_subpart
    
    @property
    def get_seg_feats(self):
        return self._seg_feats_subpart
    
    @property
    def get_seg_feats_obj(self): 
        return self._seg_feats_obj
    
    @property
    def get_seg_feats_part(self): 
        return self._seg_feats_part
    
    @property
    def get_seg_feats_subpart(self): 
        return self._seg_feats_subpart

    @property
    def get_shading_normals(self):
        return self._normal1

    def get_normal(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals

    def get_normal_2(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        #splat2world = self.get_covariance(scaling_modifier)
        #normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(self._normal1, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)
    
    @property
    def get_asg(self):
        return self._indirect_asg
    
    def render_env_map(self, H=512, render_approx=False):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        if render_approx:
            return {'env1': self.approx_env(directions1, mode="pure_env"), 'env2': self.approx_env(directions2, mode="pure_env")}
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}
    
    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    def init_shading_normals(self):
        with torch.no_grad():
            # Get normalized rotation representation
            #rotation = self.get_rotation  # (N, 4) quaternion normalized
            # Build covariance transform (without scaling modifier)
            splat2world = self.get_covariance(scaling_modifier=1.0)  # (N, 4, 4)
            # Extract geometric normal (3rd column of rotation matrix)
            normals = splat2world[:, 2, :3]  # (N, 3)
            # Normalize for safety
            normals = torch.nn.functional.normalize(normals, dim=-1)

            # Initialize learnable delta normal
            self._normal1 = nn.Parameter(normals.clone().detach().requires_grad_(True))
        #print(self._normal1.shape, self._normal1.max(), self._normal1.min())

    @property   
    def get_envmap(self):
        if self.novel_envs and self.use_nv_env:
            if self.use_nv_env > 0:
                novel_env = self.novel_envs[self.use_nv_env]
            else:
                novel_env = np.random.choice(self.novel_envs)
            self.use_nv_env = 0
            return novel_env
        return self.env_map
    
    @property   
    def get_envmap_2(self): 
        return self.env_map_2
    
    def set_envmap(self, envmap_path, isBlender=True):
        self.env_map = EnvLight(path=envmap_path, device='cuda', trainable=False).cuda()
        if isBlender:
            def rotation_y(theta_deg, device="cuda"):
                import math
                theta = math.radians(theta_deg)

                c = math.cos(theta)
                s = math.sin(theta)

                return torch.tensor([
                    [ c, 0, s],
                    [ 0, 1, 0],
                    [-s, 0, c]
                ], dtype=torch.float32, device=device)

            transform = torch.tensor([
                [0, -1, 0], 
                [0, 0, 1], 
                [-1, 0, 0]
            ], dtype=torch.float32, device="cuda")

            rot = rotation_y(90.0)
            transform = rot @ transform

            self.env_map.set_transform(transform)
    
    @property   
    def get_refl_strength_to_total(self):
        refl = self.get_refl
        return (refl>0.1).sum() / refl.shape[0]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        refl = self.inverse_refl_activation(torch.ones_like(opacities).cuda() * self.init_refl_value)
        refl_strength = refl.cuda()

        roughness = self.inverse_roughness_activation(torch.ones_like(opacities).cuda() * self.init_roughness_value)
        roughness = roughness.cuda()

        def initialize_ori_color(point_cloud, init_color= 0.5, noise_level=0.05):
            base_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            ori_color = base_color + noise
            ori_color = torch.clamp(ori_color, 0.0, 1.0)
            return ori_color
        
        ori_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._refl_strength = nn.Parameter(refl_strength.requires_grad_(True)) # Metallic
        self._ori_color = nn.Parameter(ori_color.requires_grad_(True)) # Albedo
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        normals1 = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals1)
        self._normal1 = nn.Parameter(torch.from_numpy(normals1).to(self._xyz.device).requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(False))

        self._seg_ids_obj = torch.zeros_like(opacities).cuda()
        self._seg_ids_part = torch.zeros_like(opacities).cuda()
        self._seg_ids_subpart = torch.zeros_like(opacities).cuda()

        self._seg_feats_obj = torch.zeros_like(opacities).cuda()
        self._seg_feats_part = torch.zeros_like(opacities).cuda()
        self._seg_feats_subpart = torch.zeros_like(opacities).cuda()
        #self._seg_feats = torch.zeros_like(opacities).cuda()

        self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
        self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},     
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"}     
        ]

        self._normal1.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._refl_strength], 'lr': training_args.refl_strength_lr, "name": "refl_strength"}, # Metallic
            {'params': [self._ori_color], 'lr': training_args.ori_color_lr, "name": "ori_color"}, # Albedo
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._normal1], 'lr': 0.006, "name": "normal1"}, # Shading normals
            {'params': [self._normal2], 'lr': 0.0, "name": "normal2"},
        ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.env_scheduler_args = None
        if training_args.envmap_cubemap_lr_final:
            self.env_scheduler_args = get_expon_lr_func(lr_init=training_args.envmap_cubemap_lr,
                                                        lr_final=training_args.envmap_cubemap_lr_final,
                                                        lr_delay_mult=1.0,
                                                        max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, initial_stage, init_until_iter):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if (param_group["name"] == "env" or param_group["name"] == "env2") and self.env_scheduler_args and not initial_stage:
                lr = self.env_scheduler_args(iteration - init_until_iter)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz','nx2', 'ny2', 'nz2']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('refl_strength') 
        l.append('roughness') 
        for i in range(self._ori_color.shape[1]):
            l.append('ori_color_{}'.format(i))


        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        for i in range(self._seg_ids_obj.shape[1]):
            l.append(f'seg_obj_id_{i}')
        for i in range(self._seg_ids_part.shape[1]):
            l.append(f'seg_part_id_{i}')
        for i in range(self._seg_ids_subpart.shape[1]):
            l.append(f'seg_subpart_id_{i}')

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        refl_strength = self._refl_strength.detach().cpu().numpy()        
        roughness = self._roughness.detach().cpu().numpy()    
        ori_color = self._ori_color.detach().cpu().numpy()      
        
        normals1 = self._normal1.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() 

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        seg_ids_obj = self._seg_ids_obj.cpu().numpy()
        seg_ids_part = self._seg_ids_part.cpu().numpy()
        seg_ids_subpart = self._seg_ids_subpart.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals1, normals2, f_dc, f_rest, opacities, refl_strength, roughness, ori_color, scale, rotation, seg_ids_obj, seg_ids_part, seg_ids_subpart), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map is not None:
            save_path = path.replace('.ply', '1.map')
            torch.save(self.env_map.state_dict(), save_path)

        if self.env_map_2 is not None:
            save_path = path.replace('.ply', '2.map')
            torch.save(self.env_map_2.state_dict(), save_path)
                

    def reset_opacity0(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]


    def reset_refl(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = self._refl_strength[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl_strength")
        if "refl_strength" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl_strength"]


    def dist_rot(self): 
        REFL_MSK_THR = self.refl_msk_thr
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[refl_msk] = rot[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]

    def dist_albedo(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._ori_color.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "ori_color")
        if "ori_color" not in optimizable_tensors: return
        self._ori_color = optimizable_tensors["ori_color"]

    def dist_color(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_refl_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, REFL_MSK_THR=0.02, ROUGH_MSK_THR=0.1, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        REFL_MSK_THR = self.refl_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(refl_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_refl_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        # 重置 features_dc
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        # 重置 features_rest
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        # 将新的features_dc和features_rest替换到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        # 更新active_sh_degree
        self.active_sh_degree = 0

        # 更新类中的属性
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_ori_color(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._ori_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        ori_color_new = base_color + noise
        ori_color_new = torch.clamp(ori_color_new, 0.0, 1.0)
        
        # 将重置后的 ori_color 更新到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(ori_color_new), "ori_color")
        if "ori_color" in optimizable_tensors:
            self._ori_color = optimizable_tensors["ori_color"]

    def reset_refl_strength(self, reset_value=0.01):
        refl_strength_new = torch.full_like(self._refl_strength, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(refl_strength_new), "refl_strength")
        if "refl_strength" in optimizable_tensors:
            self._refl_strength = optimizable_tensors["refl_strength"]
    
    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]


    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # # 
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        refl_strength = np.asarray(plydata.elements[0]["refl_strength"])[..., np.newaxis] # #

        ori_color = np.stack((np.asarray(plydata.elements[0]['ori_color_0']),
                              np.asarray(plydata.elements[0]['ori_color_1']),
                              np.asarray(plydata.elements[0]['ori_color_2'])),  axis=1)
        
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #

        normal1 = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                        np.asarray(plydata.elements[0]["ny2"]),
                        np.asarray(plydata.elements[0]["nz2"])),  axis=1)


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # #
        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1)  and os.path.exists(map_path2):
                self.env_map = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map.load_state_dict(torch.load(map_path1))
                self.env_map.build_mips()
                self.env_map_2 = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map_2.load_state_dict(torch.load(map_path2))
                self.env_map_2.build_mips()
        else:
            if args.envmap_path:
                print(f"Loading environment at {args.envmap_path}")
                map_path = args.envmap_path
            else:
                print(f"Loading environment at {path.replace('.ply', '.hdr')}")
                map_path = path.replace('.ply', '.hdr')
            self.env_map = EnvLight(path=map_path, device='cuda', trainable=False).cuda()


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._refl_strength = nn.Parameter(torch.tensor(refl_strength, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._ori_color = nn.Parameter(torch.tensor(ori_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #

        self._normal1 = nn.Parameter(torch.tensor(normal1, dtype=torch.float, device="cuda").requires_grad_(True))       # #
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(False))       # #

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Segmentation masks
        ply_props = plydata.elements[0].data.dtype.names
        seg_obj_id_cols = [name for name in ply_props if name.startswith("seg_obj_id_")]
        if len(seg_obj_id_cols) > 0:
            seg_obj_id_cols = sorted(seg_obj_id_cols, key=lambda x: int(x.split('_')[-1]))
            seg_ids = np.stack([np.asarray(plydata.elements[0][name]) for name in seg_obj_id_cols], axis=1)
            self._seg_ids_obj = torch.tensor(seg_ids, dtype=torch.float, device="cuda")
        else:
            self._seg_ids_obj = torch.zeros((self._xyz.shape[0], 1), device="cuda", dtype=torch.float)
        seg_part_id_cols = [name for name in ply_props if name.startswith("seg_part_id_")]
        if len(seg_part_id_cols) > 0:
            seg_part_id_cols = sorted(seg_part_id_cols, key=lambda x: int(x.split('_')[-1]))
            seg_ids = np.stack([np.asarray(plydata.elements[0][name]) for name in seg_part_id_cols], axis=1)
            self._seg_ids_part = torch.tensor(seg_ids, dtype=torch.float, device="cuda")
        else:
            self._seg_ids_part = torch.zeros((self._xyz.shape[0], 1), device="cuda", dtype=torch.float)
        seg_subpart_id_cols = [name for name in ply_props if name.startswith("seg_subpart_id_")]
        if len(seg_subpart_id_cols) > 0:
            seg_subpart_id_cols = sorted(seg_subpart_id_cols, key=lambda x: int(x.split('_')[-1]))
            seg_ids = np.stack([np.asarray(plydata.elements[0][name]) for name in seg_subpart_id_cols], axis=1)
            self._seg_ids_subpart = torch.tensor(seg_ids, dtype=torch.float, device="cuda")
        else:
            self._seg_ids_subpart = torch.zeros((self._xyz.shape[0], 1), device="cuda", dtype=torch.float)
        # Segmentation features
        seg_feats_obj_path = os.path.join(args.model_path, "seg_feats_obj.pt")
        seg_feats_part_path = os.path.join(args.model_path, "seg_feats_part.pt")
        seg_feats_subpart_path = os.path.join(args.model_path, "seg_feats_subpart.pt")
        if os.path.exists(seg_feats_obj_path) and os.path.exists(seg_feats_part_path) and os.path.exists(seg_feats_subpart_path):
            self._seg_feats_obj = torch.load(seg_feats_obj_path).float().cuda()
            self._seg_feats_part = torch.load(seg_feats_part_path).float().cuda()
            self._seg_feats_subpart = torch.load(seg_feats_subpart_path).float().cuda()
            #self._seg_feats = torch.cat([self._seg_feats_obj, self._seg_feats_part, self._seg_feats_subpart], dim=0)
        else:
            self._seg_feats_obj = torch.zeros((0, 1), dtype=torch.float, device="cuda")
            self._seg_feats_part = torch.zeros((0, 1), dtype=torch.float, device="cuda")
            self._seg_feats_subpart = torch.zeros((0, 1), dtype=torch.float, device="cuda")
            #self._seg_feats = torch.zeros((0, 1), dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "env2": continue   # #
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #

        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "env2": continue   # #
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_refl_strength, new_roughness, new_ori_color, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2):
        d = {"xyz": new_xyz,
             
        "refl_strength": new_refl_strength,    # #
        "roughness": new_roughness,    # #
        "ori_color": new_ori_color,    # #

        "normal1" : new_normal1,       # #
        "normal2" : new_normal2,       # #

        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #

        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_refl_strength = self._refl_strength[selected_pts_mask].repeat(N,1)   # #
        new_ori_color = self._ori_color[selected_pts_mask].repeat(N,1)   # #
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #

        new_normal1 = self._normal1[selected_pts_mask].repeat(N,1)        # #
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)       # #

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_refl_strength, new_roughness, new_ori_color, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal1, new_normal2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_refl_strength = self._refl_strength[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_ori_color = self._ori_color[selected_pts_mask]   # #

        new_normal1 = self._normal1[selected_pts_mask]       # #
        new_normal2 = self._normal2[selected_pts_mask]       # #

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_refl_strength, new_roughness, new_ori_color, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    # From https://github.com/chensjtu/GaussianObject/blob/main/scene/gaussian_model.py
    def remove_outliers(self, opt, step, linear=False, removing_ratio=0, remaining_rate=0.0):
        """Assume as Gaussian distribution
        removing_ratio: the removing_ratio of removing points
        remain_rate: the ratio of remaining points in ratio
        """
        xyz = self.get_xyz.detach()
        num_points = xyz.shape[0]
        if not linear:
            lambda_sigma = float(z_score_from_percentage(removing_ratio)) if removing_ratio > 0 else 1 # default 1
        else: 
            # in this case, we believe that the pointcloud are more and more dense,
            # so the std_nearest_k_distance + mean_nearest_k_distance are smaller and smaller
            # thus the lambda_sigma should be bigger and bigger, from 1 -> z_score_from_percentage(1)
            iter_start = opt.densify_from_iter
            iter_end = opt.densify_until_iter
            init_lambda_sigma = 1
            final_lambda_sigma = float(z_score_from_percentage(1))
            lambda_sigma = init_lambda_sigma + (final_lambda_sigma - init_lambda_sigma) * (step - iter_start) / (iter_end - iter_start)
        nearest_k_distance = p3dops.knn_points(
            xyz.unsqueeze(0),
            xyz.unsqueeze(0),
            K=int(num_points**0.5),
        ).dists
        mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
        mask = nearest_k_distance.mean(dim = -1) >= (mean_nearest_k_distance + lambda_sigma * std_nearest_k_distance)
        mask = mask.squeeze()

        if remaining_rate > 0: # since the gs cannot generate point from None, we may need to keep some points incase of empty
            num_remove_points = mask.sum()
            num_remain_points = int(num_remove_points * remaining_rate)
            remove_indices = torch.where(mask)[0]
            # Randomly select indices to keep
            keep_indices = remove_indices[torch.randperm(remove_indices.size(0))[:num_remain_points]]
            # Update the mask: set selected points to False (to keep)
            mask[keep_indices] = False

        self.prune_points(mask)
        torch.cuda.empty_cache()

    # #
    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state   