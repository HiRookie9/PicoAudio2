from typing import Sequence
import random
from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.schedulers as noise_schedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

import numpy as np
from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.content_encoder.caption_encoder import ContentEncoder
from models.common import LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase
from utils.torch_utilities import (
    create_alignment_path, create_mask_from_length, loss_with_mask,
    trim_or_pad_length
)


class DiffusionMixin:
    def __init__(
        self,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True,
        cfg_drop_ratio: float = 0.2,

    ) -> None:
        self.noise_scheduler_name = noise_scheduler_name
        self.snr_gamma = snr_gamma
        self.classifier_free_guidance = classifier_free_guidance
        self.cfg_drop_ratio = cfg_drop_ratio
        self.noise_scheduler = noise_schedulers.DDPMScheduler.from_pretrained(
            self.noise_scheduler_name, subfolder="scheduler"
        )

    def compute_snr(self, timesteps) -> torch.Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device
                                                    )[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[...,
                                                                          None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma)**2
        return snr

    def get_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        training: bool = True
    ) -> torch.Tensor:
        if training:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size, ),
                device=device
            )
        else:
            # validation on half of the total timesteps
            timesteps = (self.noise_scheduler.config.num_train_timesteps //
                         2) * torch.ones((batch_size, ),
                                         dtype=torch.int64,
                                         device=device)

        timesteps = timesteps.long()
        return timesteps

    def get_target(
        self, latent: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the target for loss depending on the prediction type
        """
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latent, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )
        return target

    def loss_with_snr(
        self, pred: torch.Tensor, target: torch.Tensor,
        timesteps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask)
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)],
                            dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask, reduce=False) * mse_loss_weights
            loss = loss.mean()
        return loss


class AudioDiffusion(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    DiffusionMixin
):  
    """
    Args:
        autoencoder (AutoEncoderBase): Pretrained autoencoder module VAE(frozen).
        content_encoder (ContentEncoder): Encodes TCC and TDC information.
        backbone (nn.Module): Main denoising network.
        frame_resolution (float): Resolution for audio frames.
        noise_scheduler_name (str): Noise scheduler identifier.
        snr_gamma (float, optional): SNR gamma for noise scheduler.
        classifier_free_guidance (bool): Enable classifier-free guidance.
        cfg_drop_ratio (float): Ratio for randomly dropping context for classifier-free guidance.
    """
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        frame_resolution:float,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True,
        cfg_drop_ratio: float = 0.2,
    ):
        nn.Module.__init__(self)
        DiffusionMixin.__init__(
            self, noise_scheduler_name, snr_gamma, classifier_free_guidance, cfg_drop_ratio
        )
        
        self.autoencoder = autoencoder
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.content_encoder = content_encoder
        self.backbone = backbone
        self.frame_resolution = frame_resolution
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self, content: list[Any], condition: list[Any], task: list[str],
        waveform: torch.Tensor, waveform_lengths: torch.Tensor, **kwargs
    ):  
        """
        Training forward pass.

        Args:
            content (list[Any]): List of content dicts for each sample.
            condition (list[Any]): Conditioning information (unused here).
            task (list[str]): List of task types.
            waveform (Tensor): Batch of waveform tensors.
            waveform_lengths (Tensor): Lengths for each waveform sample.

        Returns:
            dict: Dictionary containing the diffusion loss.
        """
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )
        # content(non_time_aligned_content) for TCC and time_aligned_content for TDC
        content, content_mask, onset, _= self.content_encoder.encode_content(
            content, task, device=device
        )

        # prepare latent and diffusion-related noise
        time_aligned_content = onset.permute(0,2,1)
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0
                time_aligned_content[mask_indices] = 0

        batch_size = latent.shape[0]
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

        # Denoising prediction
        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)
        return {
            "diff_loss": diff_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        disable_progress: bool = True,
        num_samples_per_content: int = 1,
        **kwargs
    ):
        """
        Inference/generation method for audio diffusion.

        Args:
            content (list[Any]): List of content dicts.
            condition (list[Any]): Conditioning info (unused here).
            task (list[str]): List of task types.
            scheduler (SchedulerMixin): Scheduler for timesteps and noise.
            num_steps (int): Number of denoising steps.
            guidance_scale (float): Classifier-free guidance scale.
            guidance_rescale (float): Rescale factor for guidance.
            disable_progress (bool): Disable progress bar.
            num_samples_per_content (int): How many samples to generate per content.

        Returns:
            waveform (Tensor): Generated waveform.
        """
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(content) * num_samples_per_content

        if classifier_free_guidance:
            content, content_mask, onset, length_list = self.encode_content_classifier_free(
                content, task, num_samples_per_content
            )
        else:
            content, content_mask, onset, length_list = self.content_encoder.encode_content(
            content, task, device=device
        )
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps


        # prepare input latent and context for the backbone
        shape = (batch_size, 128, onset.shape[2])  # 128 for StableVAE channels
        time_aligned_content = onset.permute(0,2,1)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )
        
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma
        latent_mask = torch.full((batch_size, onset.shape[2]), False, device=device)
        
        for i, length in enumerate(length_list):
        # Set latent mask True for valid time steps for each sample
            latent_mask[i, :length] = True
        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        if classifier_free_guidance:
            uncond_time_aligned_content = torch.zeros_like(
                time_aligned_content
            )
            time_aligned_content = torch.cat(
                [uncond_time_aligned_content, time_aligned_content]
            )
            latent_mask = torch.cat(
                [latent_mask, latent_mask.detach().clone()]
            )

        # iteratively denoising

        for i, timestep in enumerate(timesteps):

            latent_input = torch.cat(
                [latent, latent]
            ) if classifier_free_guidance else latent
            latent_input = scheduler.scale_model_input(latent_input, timestep)

            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=content,
                context_mask=content_mask,
            )

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_content - noise_pred_uncond
                )
                if guidance_rescale != 0.0:
                    noise_pred = self.rescale_cfg(
                        noise_pred_content, noise_pred, guidance_rescale
                    )
            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i+1) % scheduler.order == 0):
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)
        return waveform

    def encode_content_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        num_samples_per_content: int = 1
    ):
        device = self.dummy_param.device

        content, content_mask, onset, length_list = self.content_encoder.encode_content(
            content, task, device=device
        )
        content = content.repeat_interleave(num_samples_per_content, 0)
        content_mask = content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # get unconditional embeddings for classifier free guidance
        uncond_content = torch.zeros_like(content)
        uncond_content_mask = content_mask.detach().clone()

        uncond_content = uncond_content.repeat_interleave(
            num_samples_per_content, 0
        )
        uncond_content_mask = uncond_content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        content = torch.cat([uncond_content, content])
        content_mask = torch.cat([uncond_content_mask, content_mask])

        return content, content_mask, onset, length_list
    
    def rescale_cfg(
        self, pred_cond: torch.Tensor, pred_cfg: torch.Tensor,
        guidance_rescale: float
    ):
        """
        Rescale `pred_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = pred_cond.std(
            dim=list(range(1, pred_cond.ndim)), keepdim=True
        )
        std_cfg = pred_cfg.std(dim=list(range(1, pred_cfg.ndim)), keepdim=True)

        pred_rescaled = pred_cfg * (std_cond / std_cfg)
        pred_cfg = guidance_rescale * pred_rescaled + (
            1 - guidance_rescale
        ) * pred_cfg
