import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.io import read_image
from tqdm import tqdm
from utils.loss_utils import l1_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

# from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning 

#os.environ["HUGGINGFACE_HOME"] = os.path.abspath("./hugging_face/hub")
#os.environ["CLIP_HOME"] = os.path.abspath("./hugging_face/clip")
#os.environ['TORCH_HOME'] = os.path.abspath("./hugging_face/torch")
#stabilityai/stable-diffusion-2-1-base original

class GuidanceModel(nn.Module):
    def __init__(self, device, gd_model_id="Manojb/stable-diffusion-2-1-base", loss_type="sds", is_latent=True, dtype=torch.float32, text_cond=True):
        super().__init__()
        self.device = device
        self.gd_model_id = gd_model_id
        self.loss_type = loss_type
        self.dtype = dtype
        self.is_latent = is_latent

        if text_cond:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.gd_model_id, subfolder="tokenizer", torch_dtype=self.dtype)
            self.text_encoder = CLIPTextModel.from_pretrained(self.gd_model_id, subfolder="text_encoder", torch_dtype=self.dtype)
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder.requires_grad_(False)
        else:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.gd_model_id, subfolder="image_encoder", torch_dtype=self.dtype)
            self.feature_extractor =  CLIPImageProcessor.from_pretrained(self.gd_model_id, subfolder="feature_extractor", torch_dtype=self.dtype)
            self.image_encoder = self.image_encoder.to(device)
            self.image_encoder.requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(self.gd_model_id, subfolder="vae", torch_dtype=self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(self.gd_model_id, subfolder="unet", torch_dtype=self.dtype)
        self.scheduler = DDIMScheduler.from_pretrained(self.gd_model_id, subfolder="scheduler", torch_dtype=self.dtype)

        self.unet = self.unet.to(device)
        self.unet.requires_grad_(False)
        self.vae = self.vae.to(device)
        self.vae.requires_grad_(False)
        self.unet_cross_attention_kwargs = {'scale': 0}

        if self.loss_type == "vsd":
            self.unet_phi = self.extract_lora_diffusers(self.unet)
            self.unet_phi = self.unet_phi.to(self.device)
            self.phi_params = list(self.unet_phi.parameters())
            self.unet_phi_cross_attention_kwargs = {'scale': 1.0}

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.embeddings = None

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)

    def extract_lora_diffusers(self, unet):
        ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
        ### begin lora
        # Set correct lora layers
        unet_lora_attn_procs = {}
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = LoRAAttnProcessor

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(self.device)
        unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(unet.attn_processors)

        unet.requires_grad_(False)
        for param in unet_lora_layers.parameters():
            param.requires_grad_(True)
        return unet
    
    @torch.no_grad()
    def inference(self, input, current_iter, max_iter, num_inference_steps=10, guidance_scale=3):
        # Modified from huggingface https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline
        if not self.is_latent:
            latents = self.encode_input(input)
        else:
            latents = input
        self.scheduler.set_timesteps(num_inference_steps)

        #tmin = 1.0 - (current_iter / max_iter)
        #tmin = 0.0 + (current_iter / max_iter)
        tmin = 0.02
        tmax = 0.98
        #tmax = 1.0
        #t = 0.66
        t = torch.FloatTensor([0]).uniform_(tmin, tmax).cuda()#.item()

        # Uniformly sampled timestep
        init_step = int(num_inference_steps * t)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            # predict the noise residual
            noise_pred = self.predict_noise(self.unet, latents, self.embeddings, t.cuda().unsqueeze(0), guidance_scale=guidance_scale, cross_attention_kwargs=self.unet_cross_attention_kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        decoded_img = self.decode_latent(latents)

        return decoded_img, self.scheduler.timesteps[init_step]
    
    @torch.no_grad()
    def embed_prompts(self, prompt, batch_size=1):
        # Create tokens
        pos_tokens = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        neg_tokens = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        # Generate embeddings
        pos_embeds = self.text_encoder(pos_tokens.input_ids.to(self.device))[0]
        neg_embeds = self.text_encoder(neg_tokens.input_ids.to(self.device))[0]

        self.embeddings = torch.cat([neg_embeds.expand(batch_size, -1, -1), pos_embeds.expand(batch_size, -1, -1)])

    @torch.no_grad()
    def embed_image(self, image):
        # Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_image_variation.py#L356
        image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False)
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, 1, 1)
        image_embeddings = image_embeddings.view(bs_embed * 1, seq_len, -1)

        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        self.embeddings = image_embeddings

    def predict_noise(self, unet, noisy_latents, embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}):
        if guidance_scale == 1:
            batch_size = noisy_latents.shape[0]
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample
        else:
            latent_model_input = torch.cat([noisy_latents] * 2)
            t_in = torch.cat([t] * 2)
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t_in, encoder_hidden_states=embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred.float()
        return noise_pred
    
    def loss(self, input, current_iter=None, max_iter=None, render_dir=None, step_ratio=None, guidance_scale=7.5):
        if self.loss_type == "recon":
            return self.recon_loss(input, current_iter, max_iter, render_dir, step_ratio=step_ratio, guidance_scale=guidance_scale)
        else:
            return self.sds_vsd_loss(input, step_ratio=step_ratio, guidance_scale=guidance_scale)
    
    def recon_loss(self, input, current_iter, max_iter, render_dir=None, step_ratio=None, guidance_scale=3):
        batch_size = input.shape[0]
        target, t = self.inference(input, current_iter, max_iter, guidance_scale=guidance_scale)
        #w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        w = 1.0
        input = F.interpolate(input, (512, 512), mode="bilinear", align_corners=False)

        if render_dir:
            if (current_iter % 500 == 0):
                save_image(target.cpu(), os.path.join(render_dir, f"{str(current_iter).zfill(8)}_fake_diff_denoised.png"))
        
        loss = w*(l1_loss(target, input) + self.lpips(2 * target - 1, 2 * input - 1))
        return loss

    def sds_vsd_loss(self, input, step_ratio=None, guidance_scale=7.5):
        batch_size = input.shape[0]
        if not self.is_latent:
            latents = self.encode_input(input)
        else:
            latents = input
        if step_ratio is not None:
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        # Diffusing process
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # Noise prediction
        with torch.no_grad():
            noise_pred = self.predict_noise(self.unet, noisy_latents, self.embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=self.unet_cross_attention_kwargs)

        # VSD
        if self.loss_type == "vsd":
            with torch.no_grad():
                noise_pred_phi = self.predict_noise(self.unet_phi, noisy_latents, self.embeddings, t, guidance_scale=1.0, cross_attention_kwargs=self.unet_phi_cross_attention_kwargs)
            grad = (noise_pred - noise_pred_phi.detach())
        # SDS
        else:
            grad = (noise_pred - noise)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        grad = w* grad
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum")

        return loss

    def loss_phi(self, input):
        # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114
        # predict the noise residual with unet
        batch_size = input.shape[0]
        if not self.is_latent:
            latents_phi = self.encode_input(input)
        else:
            latents_phi = input
        t_phi = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        noise_phi = torch.randn_like(latents_phi)
        noisy_latents_phi = self.scheduler.add_noise(latents_phi, noise_phi, t_phi)

        loss_fn = nn.MSELoss()
        noise_pred = self.predict_noise(self.unet_phi, noisy_latents_phi.detach(), self.embeddings, t_phi, guidance_scale=1, cross_attention_kwargs=self.unet_phi_cross_attention_kwargs)
        loss_phi = loss_fn(noise_pred, noise_phi)

        return loss_phi
    
    def encode_input(self, input):
        img = F.interpolate(input, (512, 512), mode="bilinear", align_corners=False)
        img = 2 * img - 1
        latent = self.vae.encode(img).latent_dist.mode() * self.vae.config.scaling_factor
        return latent
    
    def decode_latent(self, input):
        latents_vsd = F.interpolate(input, (64, 64), mode="bilinear", align_corners=False)
        tmp_latents = 1 / self.vae.config.scaling_factor * latents_vsd.clone().detach()
        with torch.no_grad():
            image = self.vae.decode(tmp_latents).sample.to(torch.float32)
        image = (image/2+0.5).clamp(0, 1)
        return image