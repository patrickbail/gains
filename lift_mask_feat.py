import os
import clip
import torch
from functools import partial
from segment_anything import SamAutomaticMaskGenerator, build_sam

from scene import Scene
from utils.render_utils import generate_path
from gaussian_renderer import GaussianModel
from gaussian_renderer import render_pbr
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def extract_features(image, masks, clip_model, preprocess_clip, dinov2_model, preprocess_dinov2):
    device = "cuda"
    num_masks = len(masks)
    feat_dim = 1280  # or compute dynamically if needed
    features = torch.zeros((num_masks, feat_dim), device=device)

    for i, m in enumerate(masks):
        if m.sum() == 0:
            continue

        masked_img = image.clone()
        masked_img[~m] = 0

        pil_img = masked_img.permute(2, 1, 0).cuda()

        clip_in = preprocess_clip(pil_img).to(device)
        with torch.no_grad():
            clip_feat = clip_model.encode_image(clip_in)

        dino_in = preprocess_dinov2(pil_img).to(device)
        with torch.no_grad():
            dino_feat = dinov2_model(dino_in)

        feat = torch.cat([clip_feat.flatten(), dino_feat.flatten()]).float()
        features[i] = feat

    return features

def classify_masks_by_inclusion(masks, threshold=0.9, object_area_thresh=0.4, ignore_area_thresh=0.005, alpha_mask=None):
    device = masks.device
    masks = masks.float()  # (N, H, W)
    num_masks, H, W = masks.shape

    # Restrict valid region if alpha_mask provided
    if alpha_mask is not None:
        alpha_mask = alpha_mask.to(device).float()
        masks = masks * alpha_mask.unsqueeze(0)
        valid_area = alpha_mask.sum() + 1e-6
    else:
        valid_area = float(H * W)

    includes_count = torch.zeros(num_masks, device=device)
    included_by_count = torch.zeros(num_masks, device=device)

    areas = masks.sum((1, 2)) + 1e-6  # mask areas (inside valid region if alpha_mask used)

    for i in range(num_masks):
        mask_i = masks[i]
        area_i = areas[i]
        if area_i == 0:
            continue

        # Compute intersection with all other masks
        inter = (mask_i.unsqueeze(0) * masks).sum((1, 2))

        inc_i_j = inter / areas  # how much of j is inside i
        inc_j_i = inter / area_i # how much of i is inside j

        includes_count[i] = (inc_i_j > threshold).float().sum() - 1
        included_by_count[i] = (inc_j_i > threshold).float().sum() - 1

    mask_types = []
    rel_areas = areas / valid_area

    for i in range(num_masks):
        if includes_count[i] > 0 and included_by_count[i] == 0:
            mask_type = "object"
        elif includes_count[i] > 0 and included_by_count[i] > 0:
            mask_type = "part"
        else:
            mask_type = "subpart"

        # Area-based overrides
        if rel_areas[i] >= object_area_thresh:
            mask_type = "object"
        elif (mask_type == "subpart" or mask_type == "part") and rel_areas[i] < ignore_area_thresh:
            mask_type = "ignore"

        mask_types.append(mask_type)

    # Ensure large regions always count as objects
    for i, area in enumerate(rel_areas):
        if area >= object_area_thresh:
            mask_types[i] = "object"

    return mask_types

def compute_inclusion_matrix(masks, threshold=0.9):
    N = masks.shape[0]
    includes = torch.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            inter = torch.logical_and(masks[i], masks[j]).sum()
            size_j = masks[j].sum()
            if size_j > 0 and inter / size_j >= threshold:
                includes[i, j] = True
    return includes

def classify_masks(masks, alpha_mask=None, object_area_thresh=0.9, part_area_thresh=0.4, subpart_area_thresh=0.05):
    includes = compute_inclusion_matrix(masks)
    N, H, W = masks.shape
    types = []
    
    # Determine base area references
    if alpha_mask is not None:
        reference_area = alpha_mask.sum().float()
    else:
        reference_area = torch.tensor(H * W, dtype=torch.float32)
    
    for i in range(N):
        included_by = includes[:, i].sum()
        includes_others = includes[i, :].sum()
        
        if included_by == 0 and includes_others > 0:
            mask_type = "object"
        elif included_by > 0 and includes_others > 0:
            mask_type = "part"
        elif included_by > 0 and not includes_others == 0:
            mask_type = "subpart"
        else:
            mask_type = "ignore"

        if mask_type == "object" and alpha_mask is not None:
            inter = torch.logical_and(masks[i], alpha_mask).sum().float()
            coverage = inter / (reference_area + 1e-6)
            if coverage >= object_area_thresh:
                mask_type = "ignore"

        if mask_type == "part" and alpha_mask is not None:
            inter = torch.logical_and(masks[i], alpha_mask).sum().float()
            coverage = inter / (reference_area + 1e-6)
            if coverage >= part_area_thresh:
                mask_type = "ignore"
        
        if mask_type == "part":
            mask_area = masks[i].sum().float()
            if mask_area / (reference_area + 1e-6) < subpart_area_thresh:
                mask_type = "subpart"

        types.append(mask_type)
    
    #num_objects = sum(t == "object" for t in types)
    num_parts = sum(t == "part" for t in types)
    if num_parts > 2:
        types = ["object" if t == "part" else "ignore" if t == "object" else t for t in types]
                
    return types

def preprocess_clip_tensor(image_tensor):
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0  # ensure [0,1]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    image_tensor = F.interpolate(image_tensor.float(), size=(224, 224), mode="bicubic", align_corners=False)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor

def preprocess_dinov2_tensor(image_tensor):
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0  # ensure [0,1]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    _, _, h, w = image_tensor.shape
    target_size = 256

    scale = target_size / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image_tensor = F.interpolate(image_tensor.float(), size=(new_h, new_w), mode="bicubic", align_corners=False)

    crop_size = 224
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    image_tensor = image_tensor[:, :, top:top + crop_size, left:left + crop_size]

    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor   

def merge_similar_masks(masks, iou_thresh=0.9):
    N, H, W = masks.shape
    masks = masks.clone()

    areas = masks.flatten(1).sum(1).float()

    # Compute pairwise intersection
    inter = torch.matmul(masks.flatten(1).float(), masks.flatten(1).float().T)
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / (union + 1e-6)
    
    # Track which masks to keep
    keep = torch.ones(N, dtype=torch.bool, device=masks.device)

    for i in range(N):
        if not keep[i]:
            continue
        # Find all later masks that are highly overlapping
        dupes = (iou[i] > iou_thresh) & keep
        dupes[i] = False  # exclude self
        if dupes.any():
            # Merge all duplicates into mask i
            masks[i] |= masks[dupes].any(dim=0)
            # Mark them as removed
            keep[dupes] = False

    return masks[keep]    

def lift_mask_feat(gaussians: GaussianModel, scene: Scene, background, pipe, op, args, merge_threshold = 0.4, n_frames=100, isObjectScene=False):
    render = partial(render_pbr, pipe=pipe, bg_color=background)

    viewpoint_stack = scene.getTrainCameras()
    isObjectScene = False

    stability_score_thresh = 0.9
    pred_iou_thresh = 0.94
    if isObjectScene:
        stability_score_thresh = 0.86
        pred_iou_thresh = 0.9
        print("Scene is object!")

    sam = build_sam(checkpoint=args.sam_checkpoint).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        stability_score_thresh=stability_score_thresh,
        pred_iou_thresh=pred_iou_thresh,
    )

    preprocess_clip = preprocess_clip_tensor
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to('cuda').eval()
    preprocess_dinov2 = preprocess_dinov2_tensor

    num_gaussians = gaussians._xyz.shape[0]
    objects = []

    viewpoint_stack = generate_path(viewpoint_stack, n_frames=n_frames)

    if isObjectScene:
        ignore_list = ["part", "subpart", "ignore"]
        merge_threshold = merge_threshold / 4.0
    else:
        ignore_list = ["object", "part", "ignore"]

    for t, viewpoint_cam in enumerate(viewpoint_stack):
        render_pkg = render(viewpoint_cam, gaussians, srgb=op.srgb, opt=op)
        image = (render_pkg['render_sh'].permute(1, 2, 0).detach() * 255.0).clamp(0, 255).byte()
        image = image.to("cuda")

        alpha_mask = None
        if isObjectScene:
            alpha_mask = render_pkg['rend_alpha'].squeeze(0).bool()

        results = mask_generator.generate(image.cpu().numpy())

        masks = torch.stack([torch.from_numpy(r['segmentation']).to(torch.bool).to("cuda") for r in results])
        masks = merge_similar_masks(masks)

        if isObjectScene:
            mask_types = classify_masks(masks, alpha_mask=alpha_mask)
        else:
            mask_types = classify_masks_by_inclusion(masks, alpha_mask=alpha_mask)

        feats = extract_features(image, masks, clip_model, preprocess_clip, dinov2_model, preprocess_dinov2)

        alpha_contrib = render_pkg['alpha_contrib'].cuda()

        ignore_list_2 = ignore_list
        #print(mask_types)
        if mask_types.count("subpart") < 2:
            ignore_list_2 = ["ignore"]

        for i, mask in enumerate(masks):
            if mask_types[i] in ignore_list_2:
                continue
            G_t_i = alpha_contrib[mask]
            f_t_i = feats[i]

            best_score = 0.0
            best_obj = None
            current_type = mask_types[i]
            candidates = [obj for obj in objects if obj['type'] == current_type]

            for obj in candidates:
                G_prev_j = obj['indices']
                f_prev_j = obj['feature']

                overlap = torch.isin(G_t_i, G_prev_j).sum().float() / G_t_i.numel()
                cos_sim = torch.nn.functional.cosine_similarity(f_t_i.unsqueeze(0), f_prev_j.unsqueeze(0)).item()

                score = overlap * cos_sim
                if score > best_score:
                    best_score = score
                    best_obj = obj

            if best_obj is not None and best_score >= merge_threshold:
                merged_indices = torch.unique(torch.cat([best_obj['indices'], G_t_i]))
                best_obj['indices'] = merged_indices
                n = best_obj['count']
                best_obj['feature'] = (n * best_obj['feature'] + f_t_i) / (n + 1)
                best_obj['count'] = n + 1
            else:
                objects.append({'indices': G_t_i, 'feature': f_t_i, 'count': 1, 'type': current_type})

        print(f"Frame {t}, objects: {len(objects)}")
    
    if isObjectScene:
        print("Performing post-merge on object fragments...")
        merge_sim_threshold = 0.2  # cosine similarity threshold
        merge_overlap_threshold = 0.4  # overlap fraction threshold

        merged = True
        while merged:
            merged = False
            new_objects = []
            skip = set()

            for i, obj_i in enumerate(objects):
                if i in skip or obj_i["type"] != "object":
                    continue

                merged_obj = obj_i.copy()
                for j, obj_j in enumerate(objects):
                    if j <= i or j in skip or obj_j["type"] != "object":
                        continue

                    # Compute cosine similarity between features
                    cos_sim = F.cosine_similarity(
                        merged_obj["feature"].unsqueeze(0),
                        obj_j["feature"].unsqueeze(0),
                    ).item()

                    # Compute index overlap (fraction of smaller set)
                    overlap = torch.isin(obj_j["indices"], merged_obj["indices"]).sum().float() / obj_j["indices"].numel()
                    #score = overlap * cos_sim

                    if cos_sim > merge_sim_threshold and overlap > merge_overlap_threshold:
                        # Merge them
                        merged_obj["indices"] = torch.unique(
                            torch.cat([merged_obj["indices"], obj_j["indices"]])
                        )
                        n_i, n_j = merged_obj["count"], obj_j["count"]
                        merged_obj["feature"] = (
                            n_i * merged_obj["feature"] + n_j * obj_j["feature"]
                        ) / (n_i + n_j)
                        merged_obj["count"] = n_i + n_j
                        skip.add(j)
                        merged = True

                new_objects.append(merged_obj)

            # Replace with merged list
            objects = [obj for k, obj in enumerate(new_objects) if k not in skip]

        print(f"Post-merge complete, remaining objects: {len([o for o in objects if o['type']=='object'])}")

    object_fragments = [obj for obj in objects if obj['type'] == "object"]
    part_fragments = [obj for obj in objects if obj['type'] == "part"]
    subpart_fragments = [obj for obj in objects if obj['type'] == "subpart"]
    if len(subpart_fragments) < 2:
        subpart_fragments = part_fragments
    
    seg_ids_obj = torch.zeros(num_gaussians, len(object_fragments) + 1, device="cuda")
    seg_ids_part = torch.zeros(num_gaussians, len(part_fragments) + 1, device="cuda")
    seg_ids_subpart = torch.zeros(num_gaussians, len(subpart_fragments) + 1, device="cuda")
    for obj_id, obj in enumerate(object_fragments):
        seg_ids_obj[obj['indices'], obj_id + 1] = 1
    for obj_id, obj in enumerate(part_fragments):
        seg_ids_part[obj['indices'], obj_id + 1] = 1
    for obj_id, obj in enumerate(subpart_fragments):
        seg_ids_subpart[obj['indices'], obj_id + 1] = 1

    seg_feats_obj = torch.zeros(seg_ids_obj.shape[-1], 1280, device="cuda")
    seg_feats_part = torch.zeros(seg_ids_part.shape[-1], 1280, device="cuda")
    seg_feats_subpart = torch.zeros(seg_ids_subpart.shape[-1], 1280, device="cuda")
    for obj_id, obj in enumerate(object_fragments):
        seg_feats_obj[obj_id + 1] = obj['feature']
    for obj_id, obj in enumerate(part_fragments):
        seg_feats_part[obj_id + 1] = obj['feature']
    for obj_id, obj in enumerate(subpart_fragments):
        seg_feats_subpart[obj_id + 1] = obj['feature']

    torch.save(seg_feats_obj, os.path.join(args.model_path, "seg_feats_obj.pt"))
    gaussians._seg_feats_obj = seg_feats_obj
    torch.save(seg_feats_part, os.path.join(args.model_path, "seg_feats_part.pt"))
    gaussians._seg_feats_part = seg_feats_part
    torch.save(seg_feats_subpart, os.path.join(args.model_path, "seg_feats_subpart.pt"))
    gaussians._seg_feats_subpart = seg_feats_subpart
    #gaussians._seg_feats = torch.cat([seg_feats_obj, seg_feats_part, seg_feats_subpart], dim=0)

    gaussians._seg_ids_obj = seg_ids_obj
    gaussians._seg_ids_part = seg_ids_part
    gaussians._seg_ids_subpart = seg_ids_subpart

    num_object_level = len(object_fragments)
    num_part_level = len(part_fragments)
    num_subpart_level = len(subpart_fragments)
    print(f"Total object-level masks: {num_object_level}")
    print(f"Total part-level masks: {num_part_level}")
    print(f"Total subpart-level masks: {num_subpart_level}")

    del sam
    del mask_generator
    del clip_model
    del dinov2_model

    print("Lifted segmentation masks and features to gaussians")
        