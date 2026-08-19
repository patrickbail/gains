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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], env_map=None, skip_train=False, test=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.sparse = args.sparse

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.sparse_cameras = {}
        self.isBlender = False
        self.light_rotate = False
        self.full_sparse = args.full_sparse
        self.resolution_scales = resolution_scales

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args=args, skip_train=skip_train, test=test)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if "Synthetic4Relight" in args.source_path:
                print("Found Synthetic4Relight, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](args.source_path, args.white_background, args.eval, args=args, skip_train=skip_train)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, env_map=env_map, args=args, skip_train=skip_train)
            self.isBlender = True
            self.light_rotate = True
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = []
            if not skip_train:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                if args.sparse and not args.full_sparse:
                    print("Loading Sparse Cameras")
                    if args.rand_sparse:
                        cam_subset = self.train_cameras[resolution_scale]
                        indices = random.sample(range(len(cam_subset)), args.sparse)
                    else:
                        start, end = int(args.scope[0]), int(args.scope[0]) + int(args.scope[1])
                        if args.scope[1] == -1:
                            end = len(self.train_cameras[resolution_scale])
                        cam_subset = self.train_cameras[resolution_scale][start:end]

                        # Compute the evenly spaced indices
                        indices = [round(i * (len(cam_subset) - 1) / (args.sparse - 1)) for i in range(args.sparse)] if args.sparse > 1 else [0]

                    selected_cams = [cam_subset[i] for i in indices]

                    self.sparse_cameras[resolution_scale] = selected_cams
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            gaussian_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            self.gaussians.load_ply(gaussian_path, relight=args.relight, args=args)       
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args)

    def save(self, iteration, suffix=""):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud{suffix}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSparseCameras(self, scale=1.0):
        if self.full_sparse or not (self.full_sparse and self.sparse):
            return self.getTrainCameras(scale)
        return self.sparse_cameras[scale]