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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json, cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from torchvision.utils import save_image
import torch
import torchvision.transforms as transforms

import pyexr
import imageio as imageio
from utils.graphics_utils import rgb_to_srgb
from tqdm import tqdm
from glob import glob

from scene.light import EnvLight

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    albedo: np.array
    roughness: np.array
    metallic: np.array
    mask: np.array
    mono_depth: np.array
    mono_normal: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, args):
    cam_infos = []
    if args.iid_model == "rgb2x":
        iid_folder = os.path.join(path, f"iid_npy_{args.resolution}_{args.iid_model}") # needs to be generated
    elif args.iid_model == "teamwork":
        iid_folder = os.path.join(path, f"iid_teamwork") # needs to be generated
    depths_folder = os.path.join(path, f"depth_npy_{args.resolution}") # needs to be generated
    normals_folder = os.path.join(path, f"normals_npy_{args.resolution}") # needs to be generated  
                    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1],
            ])
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path.replace('.JPG', '.jpg'))

        #if intr.model=="SIMPLE_RADIAL":
        #    image = cv2.undistort(np.array(image), K, np.array([intr.params[3], 0,0,0]))
        #    image = Image.fromarray(image.astype('uint8')).convert('RGB')

        real_im_scale = image.size[0] / width
        K[:2] *=  real_im_scale

        albedo = None
        roughness = None
        metallic = None

        if args.lambda_iid:
            if "teamwork" in iid_folder:
                    albedo_path = os.path.join(iid_folder, image_name + "_albedo.png")
                    albedo = Image.open(albedo_path).convert("RGB")
                    albedo = transforms.ToTensor()(albedo).cuda()
            else:
                albedo_path = os.path.join(iid_folder, image_name + "_albedo.npy")
                if os.path.exists(albedo_path):
                    albedo = torch.tensor(np.load(albedo_path)).permute(2, 0, 1).float()

            if not args.albedo_only_iid:
                if "marigold" in iid_folder:
                    material_path = os.path.join(iid_folder, image_name + "_material.npy") # Roughness, Metallic, None
                    if os.path.exists(material_path):
                        material = np.load(material_path)
                        roughness = torch.tensor(material[:, :, 0]).float().unsqueeze(0)
                        metallic = torch.tensor(material[:, :, 1]).float().unsqueeze(0)
                elif "rgb2x" in iid_folder:
                    roughness_path = os.path.join(iid_folder, image_name + "_roughness.npy") # Roughness
                    roughness = (torch.tensor(np.load(roughness_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0)
                    if args.include_metallic:
                        metallic_path = os.path.join(iid_folder, image_name + "_metallic.npy") # Metallic
                        metallic = (torch.tensor(np.load(metallic_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0)
                    albedo = albedo / 255.0
                    
        if albedo is not None and args.iid_model == "rgb2x" and "teamwork" not in iid_folder:
            albedo = albedo / 255.0

        depth = None
        normal = None
        if args.full_sparse:
            depth_path = os.path.join(depths_folder, image_name + "_depth.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
            if args.lambda_mono_normal:
                normal_path = os.path.join(normals_folder, image_name + "_normals.npy")
                if os.path.exists(normal_path):
                    normal = np.transpose(np.load(normal_path), (1, 2, 0))
                    normal[..., 1] *= -1
                    normal[..., 2] *= -1
                    normal = normal @ R.T

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              albedo=albedo, roughness=roughness, metallic=metallic, mask=None, 
                              mono_depth=depth, mono_normal=normal)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print('Load Ply color and normals failed, random init')
        colors = np.random.rand(*positions.shape) / 255.0
        normals = np.random.rand(*positions.shape)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, args=None, skip_train=False, test=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    indices = None
    if not test:
        if args.sparse and args.full_sparse and not test:
            start, end = int(args.scope[0]), int(args.scope[0]) + int(args.scope[1])
            if args.scope[1] == -1:
                end = len(cam_extrinsics)
            cam_extrinsics_subset = dict(list(cam_extrinsics.items())[start:end])

            # Compute the evenly spaced indices
            indices = [round(i * (len(cam_extrinsics_subset) - 1)/(args.sparse - 1)) + 1 + start for i in range(args.sparse)] if args.sparse > 1 else [0]

            cam_extrinsics = {i:cam_extrinsics_subset[i] for i in indices}
    else:
        all_extrinsics = cam_extrinsics
        all_keys = list(all_extrinsics.keys())
        start = int(args.scope[0])
        if args.scope[1] == -1:
            end = len(all_keys)
        else:
            end = start + int(args.scope[1])
        scoped_keys = all_keys[start:end]
        if args.sparse > 1:
            sparse_relative = [
                round(i * (len(scoped_keys) - 1) / (args.sparse - 1))
                for i in range(args.sparse)
            ]
        else:
            sparse_relative = [0]
        indices = [start + i for i in sparse_relative]

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), path=path, args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval and test:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in indices]
        test_cam_infos  = [c for idx, c in enumerate(cam_infos) if idx not in indices]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    del cam_infos

    if not skip_train:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(test_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    spc_ply_path = os.path.join(path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", test=False, env_map=None, args=None):
    cam_infos = []

    if args.iid_model == "rgb2x":
        iid_folder = os.path.join(path, f"iid_appearance_npy_{args.iid_model}") # needs to be generated
    elif args.iid_model == "teamwork":
        iid_folder = os.path.join(path, f"iid_teamwork") # needs to be generated
    depths_folder = os.path.join(path, "depth_npy") # needs to be generated
    normals_folder = os.path.join(path, f"normals_npy") # needs to be generated
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]

        if args.full_sparse and not test and not (args.scope[0] == 0 and args.scope[1] == -1):
            if args.sparse == -1:
                sparse = len(frames)
            else:
                sparse = args.sparse
                
            start, end = int(args.scope[0]), int(args.scope[0]) + int(args.scope[1])
            if args.scope[1] == -1:
                end = len(frames)
            frames_subset = frames[start:end]

            # Compute the evenly spaced indices
            indices = [round(i * (len(frames_subset) - 1)/(sparse - 1)) for i in range(sparse)] if sparse > 1 else [0]

            frames = [frames_subset[i] for i in indices]

        for idx, frame in enumerate(frames):
            if "tensorIR" in path:
                if env_map:
                    cam_name = os.path.join(path, "test" if test else "train", frame["file_path"][2:] + f"_{env_map}" + extension)
                else:
                    cam_name = os.path.join(path, "test" if test else "train", frame["file_path"][2:] + extension)
            elif "ref_syn" in path:
                cam_name = os.path.join(path, frame["file_path"][2:] + extension)
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            R_c2w = c2w[:3, :3].copy()
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            ### NOTE !!!!!
            # Here R has been transposed, R = w2c.T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image_name_2 = image_name
            img_id = image_path.split("/")[-2][-3:]
            if "tensorIR" in path:
                if args.full_sparse:
                    image_name_2 += f"_{int(img_id)}"
                else:
                    image_name_2 += f"_{idx}"
                image_name += f"_{img_id}"
            image = Image.open(image_path)
            with Image.open(image_path) as image:
                im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth = None
            normal = None

            if args.full_sparse:
                depth_path = os.path.join(depths_folder, image_name + "_depth.npy")
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)

                if args.lambda_mono_normal:
                    normal_path = os.path.join(normals_folder, image_name + "_normals.npy")
                    if os.path.exists(normal_path):
                        normal = np.transpose(np.load(normal_path), (1, 2, 0)) @ R_c2w.T

            mask = ((im_data[:, :, 3] == 0).astype(np.uint8) == 0)

            albedo = None
            roughness = None
            metallic = None

            if test and "tensorIR" in path:
                albedo_path = os.path.join(path, "test", frame["file_path"][2:-4] + "albedo" + extension)
                albedo = Image.open(albedo_path).convert("RGB")
                albedo = transforms.ToTensor()(albedo).cuda()

                normal_path = os.path.join(path, "test", frame["file_path"][2:-4] + "normal.png")
                normal = Image.open(normal_path)
                normal = transforms.ToTensor()(normal)[:3, :, :].cuda()
                normal = normal * 2.0 - 1.0
            elif test and "ref_syn" in path:
                albedo_path = os.path.join(path, frame["file_path"][2:] + "_albedo.png")
                albedo = Image.open(albedo_path).convert("RGB")
                albedo = transforms.ToTensor()(albedo).cuda()

                normal_path = os.path.join(path, frame["file_path"][2:] + "_normal.png")
                normal = Image.open(normal_path)
                normal = transforms.ToTensor()(normal)[:3, :, :].cuda()
                normal = normal * 2.0 - 1.0


            elif args.lambda_iid:
                if "teamwork" in iid_folder:
                    albedo_path = os.path.join(iid_folder, image_name_2 + "_albedo.png")
                    albedo = Image.open(albedo_path).convert("RGB")
                    albedo = transforms.ToTensor()(albedo).cuda()
                else:
                    albedo_path = os.path.join(iid_folder, image_name_2 + "_albedo.npy")
                    if os.path.exists(albedo_path):
                        albedo = torch.tensor(np.load(albedo_path)).permute(2, 0, 1).float().cuda()

                if not args.albedo_only_iid:
                    if "marigold" in iid_folder:
                        material_path = os.path.join(iid_folder, image_name_2 + "_material.npy") # Roughness, Metallic, None
                        if os.path.exists(material_path):
                            material = np.load(material_path)
                            roughness = torch.tensor(material[:, :, 0]).float().unsqueeze(0).cuda()
                            metallic = torch.tensor(material[:, :, 1]).float().unsqueeze(0).cuda()
                    elif "rgb2x" in iid_folder:
                        roughness_path = os.path.join(iid_folder, image_name_2 + "_roughness.npy") # Roughness
                        roughness = (torch.tensor(np.load(roughness_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0).cuda()
                        if args.include_metallic:
                            metallic_path = os.path.join(iid_folder, image_name_2 + "_metallic.npy") # Metallic
                            metallic = (torch.tensor(np.load(metallic_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0).cuda()
                        albedo = albedo / 255.0
                        
            if albedo is not None and args.iid_model == "rgb2x" and args.lambda_iid > 0 and "teamwork" not in iid_folder:
                albedo = albedo / 255.0

            # For blender datasets, we consider its camera center offset is zero (ideal camera)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            albedo=albedo, roughness=roughness, metallic=metallic, mask=mask, mono_depth=depth, 
                            mono_normal=normal))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", env_map=None, args=None, skip_train=False):
    if not skip_train:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, env_map=env_map, args=args)
    else:
        train_cam_infos = []
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, test=True, env_map=env_map, args=args)
    else:
        test_cam_infos = []

    if not skip_train:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(test_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSynthetic4RelightInfo(path, white_background, eval, debug=False, args=None, llffhold=2, skip_train=False):
    if args.gt_iid:
        print("Reading GT Training Transforms")
        cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug, test=True, args=args)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        if eval:
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            test_cam_infos = []
    else:
        if not skip_train:
            print("Reading Training Transforms")
            train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", args=args)
        else:
            train_cam_infos = []
        if eval:
            print("Reading Test Transforms")
            test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", test=True, args=args)
        else:
            test_cam_infos = []

    if not skip_train:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(test_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def load_img_rgb(path):
    
    if path.endswith(".exr"):
        exr_file = pyexr.open(path)
        img = exr_file.get()
        img[..., 0:3] = rgb_to_srgb(img[..., 0:3])
        # img[..., 0:3] = rgb_to_srgb(img[..., 0:3], clip=False)
    else:
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    return img

def load_mask_bool(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    alpha_mask = mask.astype(np.float32)/255.0
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0
    alpha_mask[alpha_mask > 0.0] = 1.0

    return mask.astype(np.float32), alpha_mask.astype(np.float32)

def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", test=False, args=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        if args.iid_model == "rgb2x":
            iid_folder = os.path.join(path, f"iid_appearance_npy_{args.iid_model}") # needs to be generated
        elif args.iid_model == "teamwork":
            iid_folder = os.path.join(path, f"iid_teamwork") # needs to be generated
        depths_folder = os.path.join(path, "depth_npy") # needs to be generated
        normals_folder = os.path.join(path, f"normals_npy") # needs to be generated
        frames = contents["frames"]

        if args.full_sparse and not test:
            start, end = int(args.scope[0]), int(args.scope[0]) + int(args.scope[1])
            if args.scope[1] == -1:
                end = len(frames)
            frames_subset = frames[start:end]

            # Compute the evenly spaced indices
            indices = [round(i * (len(frames_subset) - 1)/(args.sparse - 1)) for i in range(args.sparse)] if args.sparse > 1 else [0]

            frames = [frames_subset[i] for i in indices]
            train_indices = [start + i for i in indices]

        for idx, frame in enumerate(tqdm(frames)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            R_c2w = c2w[:3, :3].copy()
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0
            
            image = load_img_rgb(image_path)
            mask = None
            if args.gt_iid:
                mask = (image[:,:,3] != 0).astype(np.float32)
                alpha_mask = mask
            else:
                mask, alpha_mask = load_mask_bool(mask_path)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            arr = image[..., :3] * mask[..., None] + bg * (1 - mask[..., None])
            
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            albedo = None
            roughness = None
            metallic = None

            if test and args.gt_iid:
                albedo_path = os.path.join(path, frame["file_path"] + "_albedo" + extension[-4:])
                albedo = Image.open(albedo_path).convert("RGB")
                albedo = transforms.ToTensor()(albedo).cuda()

                if not args.albedo_only_iid:
                    roughness_path = os.path.join(path, frame["file_path"] + "_rough" + extension[-4:])
                    roughness = Image.open(roughness_path).convert("L")
                    roughness = transforms.ToTensor()(roughness).repeat(3, 1, 1).cuda()
            elif test:
                albedo_path = os.path.join(path, frame["file_path"] + "_albedo" + extension[-4:])
                albedo = Image.open(albedo_path).convert("RGBA")
                albedo = transforms.ToTensor()(albedo).cuda()

                roughness_path = os.path.join(path, frame["file_path"] + "_rough" + extension[-4:])
                roughness = Image.open(roughness_path).convert("L")
                roughness = transforms.ToTensor()(roughness).repeat(3, 1, 1).cuda()
            elif args.lambda_iid:
                if "teamwork" in iid_folder:
                    albedo_path = os.path.join(iid_folder, image_name + "_albedo.png")
                    albedo = Image.open(albedo_path).convert("RGB")
                    albedo = transforms.ToTensor()(albedo).cuda()
                else:
                    albedo_path = os.path.join(iid_folder, image_name + "_albedo.npy")
                    if os.path.exists(albedo_path):
                        albedo = torch.tensor(np.load(albedo_path)).permute(2, 0, 1).float().cuda()

                if not args.albedo_only_iid:
                    if "marigold" in iid_folder:
                        material_path = os.path.join(iid_folder, image_name + "_material.npy") # Roughness, Metallic, None
                        if os.path.exists(material_path):
                            material = np.load(material_path)
                            roughness = torch.tensor(material[:, :, 0]).float().unsqueeze(0).cuda()
                            metallic = torch.tensor(material[:, :, 1]).float().unsqueeze(0).cuda()
                    elif "rgb2x" in iid_folder:
                        roughness_path = os.path.join(iid_folder, image_name + "_roughness.npy") # Roughness
                        roughness = (torch.tensor(np.load(roughness_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0).cuda()
                        if args.include_metallic:
                            metallic_path = os.path.join(iid_folder, image_name + "_metallic.npy") # Metallic
                            metallic = (torch.tensor(np.load(metallic_path)).float()[..., 0:1].permute(2, 0, 1) / 255.0).cuda()
                        albedo = albedo / 255.0
                        
            if albedo is not None and args.iid_model == "rgb2x" and args.lambda_iid > 0 and "teamwork" not in iid_folder:
                albedo = albedo / 255.0

            depth = None
            normal = None

            if args.full_sparse:
                depth_path = os.path.join(depths_folder, image_name + "_depth.npy")
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)

                if args.lambda_mono_normal:
                    normal_path = os.path.join(normals_folder, image_name + "_normals.npy")
                    if os.path.exists(normal_path):
                        normal = np.transpose(np.load(normal_path), (1, 2, 0)) @ R_c2w.T

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            mono_depth=depth, mono_normal=normal, albedo=albedo, roughness=roughness, metallic=metallic, 
                            mask=alpha_mask))
    return cam_infos

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo
}