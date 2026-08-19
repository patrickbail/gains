# Modified from https://github.com/zheng95z/rgbx/tree/main/rgb2x

import os
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torchvision
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

current_directory = os.path.dirname(os.path.abspath(__file__))

import argparse
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".JPG"]

def srgb_to_linear(srgb, eps=None):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = ((200 * srgb + 11) / (211)).clamp_min(eps) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)

def run_rgb2x(args, inference_step=50):
    input_rgb_dir = args.input_dir
    output_dir = args.output_dir
    if args.resolution == 1:
        output_dir_npy = os.path.join(output_dir, f"iid_npy_{str(args.resolution)}_rgb2x")
        output_dir_vis = os.path.join(output_dir, f"iid_vis_{str(args.resolution)}_rgb2x")
    else:
        output_dir_npy = os.path.join(output_dir, "iid_appearance_npy_rgb2x")
        output_dir_vis = os.path.join(output_dir, "iid_appearance_vis_rgb2x")
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_vis, exist_ok=True)
    # Load RGB2X model
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    seed = 5555
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Image Loading
    isSRGB = "ref" not in input_rgb_dir 
    if "tensorIR" in input_rgb_dir:
        rgb_filename_list = glob(os.path.join(input_rgb_dir, "train", "train_*", "rgba.png"))
    elif "Synthetic4Relight" in input_rgb_dir:
        rgb_filename_list = glob(os.path.join(input_rgb_dir, "train", "*_rgb.exr"))
    else:
        rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
        rgb_filename_list = [
            f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST and "normal" not in f
        ]
    rgb_filename_list = sorted(rgb_filename_list)

    #required_aovs = ["albedo", "normal", "roughness", "metallic"] # irradiance
    required_aovs = ["albedo"]
    #required_aovs = ["roughness"]
    #required_aovs = ["irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        #"normal": "Camera-space Normal",
        #"roughness": "Roughness",
        #"metallic": "Metallicness",
        #"irradiance": "Irradiance (diffuse lighting)",
    }

    i = 0
    start_time = time.time()
    for rgb_path in tqdm(rgb_filename_list, desc="Material Inference", leave=True):

        if ".exr" in rgb_path:
            photo = load_exr_image(rgb_path, tonemaping=True, clamp=True)
            photo.to("cuda")
        elif (
            ".png" in rgb_path
            or ".jpg" in rgb_path
            or ".jpeg" in rgb_path
            or ".JPG" in rgb_path
        ):
            photo = load_ldr_image(rgb_path, from_srgb=isSRGB)
            photo.to("cuda")

        # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
        old_height = photo.shape[1]
        old_width = photo.shape[2]
        new_height = old_height
        new_width = old_width
        radio = old_height / old_width
        max_side = 1000
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / radio)
        else:
            new_width = max_side
            new_height = int(new_width * radio)

        if new_width % 8 != 0 or new_height % 8 != 0:
            new_width = new_width // 8 * 8
            new_height = new_height // 8 * 8

        photo = torchvision.transforms.Resize((new_height, new_width))(photo)

        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        if "tensorIR" in input_rgb_dir:
            rgb_name_base += f"_{i}"
            i += 1

        for aov_name in required_aovs:
            #print(aov_name)
            prompt = prompts[aov_name]
            generated_image = pipe(
                prompt=prompt,
                photo=photo,
                num_inference_steps=inference_step,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            generated_image = torchvision.transforms.Resize(
                (old_height, old_width)
            )(generated_image)

            # --- Save PNG (for visualization) ---
            png_save_path = os.path.join(output_dir_vis, f"{rgb_name_base}_{aov_name}.png")
            generated_image.save(png_save_path)

            npy_save_path = os.path.join(output_dir_npy, f"{rgb_name_base}_{aov_name}.npy")
            np.save(npy_save_path, generated_image)
        elapsed = time.time() - start_time
        print(f"Iteration finished. Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Multi-image Inference"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--resolution", type=int, default=0
    )
    args = parser.parse_args()
    run_rgb2x(args)