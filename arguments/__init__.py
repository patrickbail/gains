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


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_max_res = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False

        self.envmap_path = "" # for render
        self.novel_env_root_dir = "envmaps/high_res_envmaps_2k" # to provide multi-illuminated lighting conditions
        self.test_relight = "" # to provide multi-illuminated images for test

        self.sparse = 0
        self.scope = [0, -1]
        self.rand_sparse = 0
        self.seed = 0

        self.gt_iid = 0
        self.albedo_only_iid = 1
        self.include_metallic = 1

        self.lift_seg = 1 # Lift 2D SAM masks to gaussians
        self.seg_views = 100
        self.sam_checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'

        self.scene_level = 'whole'

        self.seg_test = 0

        self.step_ratio = 0.99

        self.full_sparse = 0

        self.no_diff_first = 0
        self.no_depth = 0

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 23_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.asg_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001


        self.ori_color_lr = 0.0075 
        self.refl_strength_lr =  0.005 
        self.roughness_lr =  0.005 
        #self.normal_lr = 0.006

        self.envmap_cubemap_lr = 0.01 # 0.01
        self.envmap_cubemap_lr_final = 0.0
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0

        self.lambda_normal = 0.05
        self.lambda_depth = 10.0
        self.opacity_cull = 0.1


        # initial values
        self.init_roughness_value = 0.1
        self.init_refl_value = 0.01
        self.rough_msk_thr = 0.01
        self.refl_msk_thr = 0.02

        self.enlarge_scale = 1.5


        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 16_000 

        # # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.005



        # Stage Settings
        self.initial = 0
        self.init_until_iter = 16000
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000

        self.feature_rest_from_iter = 5_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500


        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        # SRGB Transformation
        self.srgb = False
        
        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1

        self.dense_until_iter = 16_000
        self.joint_geo_mat = 0
        self.diff_at_iter = 19_000
        self.geo_diff_at_iter = 10_000

        # Naive diffusion loss on novel views
        self.lambda_diff = 0.0001 # 0.001 better albedo and relighting but worse NVS
        self.diff_loss = "sds"
        self.text_cond = 1

        # Prior: Intrinsi image decomposition
        self.lambda_iid = 2.0
        self.iid_model = "teamwork" # "rgb2x"

        # Smoothness: Total variation
        self.lambda_tv_env = 0.1

        # Smoothness: Segmentation class material consistency
        self.lambda_intra_seg = 0.1 # 1.0 for gardenspeheres

        # Stage I settings
        self.lambda_dr = 10.0
        self.lambda_pl = 0.005
        self.lambda_mono_normal = 0.25
        self.lambda_bce = 0.75

        # Misc
        self.strength_r = 1.0
        self.strength_La = 0.1

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
