from teamwork.pipelines import TeamworkPipeline
from PIL import Image
import torch
import os
from glob import glob
from tqdm import tqdm
import pyexr
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Path to the input image folder.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
args = parser.parse_args()

scenes = glob(os.path.join(args.input_dir, "*"))

#scenes = [p for p in scenes if "toaster" not in p and "teapot" not in p and "helmet" not in p]

pipe = TeamworkPipeline.from_checkpoint(
    'samsartor/teamwork-release',
    'decomposition_heterogeneous_sd3.safetensors',
    dtype=torch.bfloat16
).to('cuda')

def resize_to_multiple(image, divisor=16):
    w, h = image.size
    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def pad_to_multiple(image, divisor=16):
    w, h = image.size
    new_w = math.ceil(w / divisor) * divisor
    new_h = math.ceil(h / divisor) * divisor

    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(image, (0, 0))

    return padded

def prepare_image_for_model(image, max_w=640, divisor=16):
    w, h = image.size

    # --- Step 1: Downscale if width exceeds max_w ---
    if w > max_w:
        scale = max_w / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = image.size

    # --- Step 2: Pad to multiple of divisor (no upscaling) ---
    new_w = math.ceil(w / divisor) * divisor
    new_h = math.ceil(h / divisor) * divisor

    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(image, (0, 0))

    return padded

sphere_names = ["PXL_20210225_140630197",
"PXL_20210225_140639290",
"PXL_20210225_140653076",
"PXL_20210225_140706449",
"PXL_20210225_140713129",
"PXL_20210225_140720967",
"PXL_20210225_140734203",
"PXL_20210225_140742156"]

car_names = ["_DSC1539",
"_DSC1542",
"_DSC1546",
"_DSC1549",
"_DSC1552",
"_DSC1555",
"_DSC1560",
"_DSC1563"]

toycar_names = ["PXL_20210225_132936511",
"PXL_20210225_133019467",
"PXL_20210225_133035791",
"PXL_20210225_133044950",
"PXL_20210225_133056107",
"PXL_20210225_133109938",
"PXL_20210225_133126495",
"PXL_20210225_133139853"]

real_dict = {"gardenspheres": sphere_names, "sedan": car_names, "toycar": toycar_names}

DO_NORMAL = False

for scene in scenes:
    if "Synthetic4Relight" in args.input_dir:
        img_paths = glob(os.path.join(scene, "train", "*_rgb.exr"))
    elif "ref" in args.input_dir:
        img_paths = glob(os.path.join(scene, "train", "r_*.png"))
    elif "ref_real" in args.input_dir:
        img_paths = glob(os.path.join(scene, "images_8", "*.jpg"))
    else:
        img_paths = glob(os.path.join(scene, "train", "train_*", "rgba.png"))
    #img_paths = sorted([p for p in img_paths if "normal" not in os.path.basename(p) and "disp" not in os.path.basename(p)])
    
    if DO_NORMAL:
        args.output_dir = os.path.join(scene, "normal_teamwork")
    else:
        args.output_dir = os.path.join(scene, "iid_teamwork")
    os.makedirs(args.output_dir, exist_ok=True)

    i = 0
    for img_path in tqdm(img_paths, desc=f"Processing images for {os.path.basename(scene)}"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        #if not (filename in real_dict[os.path.basename(scene)]):
        #    continue
        if ".exr" in img_path:
            exr_np = pyexr.open(img_path).get()
            if exr_np.shape[2] == 3:
                alpha_channel = np.ones((exr_np.shape[0], exr_np.shape[1], 1), dtype=exr_np.dtype)
                exr_np = np.concatenate([exr_np, alpha_channel], axis=2)
            exr_np_255 = np.clip(exr_np * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(exr_np_255, mode="RGBA")
        else:
            image = Image.open(img_path)
            #orig_size = image.size
            #image = prepare_image_for_model(image)
            #image = image.resize((512, 512))
        if DO_NORMAL:
            generated = pipe(images={'image': image}, request=['normals'])
        else:
            generated = pipe(images={'image': image}, request=['albedo'])

        for name, out in generated.items():
            if isinstance(out, Image.Image):
                out_image = out
                #out_image = out_image.resize(orig_size, Image.LANCZOS)
            elif torch.is_tensor(out):
                # Convert torch tensor to PIL image
                if out.ndim == 3:  # C,H,W
                    out_image = Image.fromarray((out.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                elif out.ndim == 4 and out.shape[0] == 1:  # 1,C,H,W
                    out_image = Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                else:
                    raise ValueError(f"Unsupported tensor shape {out.shape} for {name}")
            else:
                try:
                    import numpy as np
                    out_image = Image.fromarray((np.array(out) * 255).astype('uint8'))
                except Exception as e:
                    print(f"Cannot save output {name}: {e}")
                    continue

            # Save image
            if DO_NORMAL:
                out_path = os.path.join(args.output_dir, os.path.basename(img_path)[:-4] + "_normal.png")
            else:
                if "tensorIR" in args.input_dir:
                    out_path = os.path.join(args.output_dir, os.path.basename(img_path)[:-4] + f"_{i}_albedo.png")
                else:
                    out_path = os.path.join(args.output_dir, os.path.basename(img_path)[:-4] + "_albedo.png")
            out_image.save(out_path)
            i += 1

print("Done")

