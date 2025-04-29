import multiprocessing
multiprocessing.set_start_method('spawn', force=True) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from typing import Callable, List, Optional, Dict, Any, Tuple
from functools import partial
from dataclasses import dataclass, field

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import einops
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf
from src.utils import create_positional_encoding
from jax.typing import DTypeLike
from einops import rearrange
from pcx.core._parameter import get as param_get  # Import get function for parameter access

import matplotlib.pyplot as plt
from datetime import datetime
import time
import math  # Import math for log10

STATUS_FORWARD = "forward"

import jax.random as jrandom
key = jrandom.PRNGKey(42)


@dataclass
class TransformerConfig:
    """Configuration for the TransformerDecoder model."""
    # Input/output dimensions
    latent_dim: int = 512
    # Image shape format: For images: (channels, height, width)
    # For videos: (frames, channels, height, width)
    image_shape: tuple = (3, 32, 32)
    
    # Video settings
    num_frames: int = 16  # Default number of frames for video
    is_video: bool = False  # Whether to use 3D positional encoding for video
    
    # Architecture settings
    hidden_size: int = 256
    num_heads: int = 8
    num_blocks: int = 3
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 256
    act_fn: Callable = jax.nn.swish # Default activation function
    
    # Patch settings
    patch_size: int = 4
    
    # Positional embedding settings
    axes_dim: list[int] = field(default_factory=lambda: [16, 16, 16])  # Default to [temporal_dim, height_dim, width_dim]
    theta: int = 10_000
    
    # Training settings
    use_noise: bool = False
    use_lower_half_mask: bool = False
    param_dtype: DTypeLike = jnp.float32
    
    def __post_init__(self):
        # Determine if we're dealing with video based on the shape of image_shape
        if len(self.image_shape) == 4:
            self.is_video = True
            self.num_frames, c, h, w = self.image_shape
        else:
            c, h, w = self.image_shape
            
        # Calculate patch dimensions
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        # For video, include temporal dimension in patch count
        if self.is_video:
            self.num_patches = self.num_frames * h_patches * w_patches
        else:
            self.num_patches = h_patches * w_patches
            
        self.patch_dim = self.patch_size * self.patch_size * c
        
        # Set positional embedding dimensions based on whether we're using video
        if not self.is_video and len(self.axes_dim) == 3:
            # If not using video but axes_dim has 3 elements, use only the last 2
            self.axes_dim = self.axes_dim[1:]
        
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = px.static(config)
        
        # Initialize random key
        key = jax.random.PRNGKey(0)

        print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
        print(f"Model initialized with {self.config.hidden_size} hidden_size, {self.config.mlp_hidden_dim} mlp_hidden_dim")
        print(f"Using {'video' if self.config.is_video else 'image'} mode with shape {self.config.image_shape}")

        # Define Vodes for predictive coding
        # Top-level latent Vode
        self.vodes = [pxc.Vode(
            energy_fn=None,
            ruleset={
                pxc.STATUS.INIT: ("h, u <- u:to_init",)
                },
            tforms={
                "to_init": lambda n, k, v, rkg: jax.random.normal(
                    px.RKG(), (config.num_patches, config.patch_dim)
                ) * 0.01 if config.use_noise else jnp.zeros((config.num_patches, config.patch_dim))
            }
        )]

        # Add a Vode for patch projection
        self.vodes.append(pxc.Vode(
                ruleset={ 
                    STATUS_FORWARD: ("h -> u",)}
            ))
        
        # Create Vodes for each transformer block output
        for _ in range(config.num_blocks):
            self.vodes.append(pxc.Vode(
                ruleset={
                    STATUS_FORWARD: ("h -> u",)}
            ))

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(config.num_blocks):
            self.transformer_blocks.append(
                pxnn.TransformerBlock(
                    input_shape=config.hidden_size,
                    hidden_dim=config.mlp_hidden_dim,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rate
                )
            )
        
        # Output Vode (sensory layer) - shape depends on whether we're handling video or images
        self.vodes.append(pxc.Vode())
        self.vodes[-1].h.frozen = True  # Freeze the output Vode's hidden state

        # Add projection layer to map from patch_dim to hidden_size
        self.patch_projection = pxnn.Linear(in_features=config.patch_dim, out_features=config.hidden_size)
        
        # Add output projection layer to map from hidden_size back to patch_dim
        self.output_projection = pxnn.Linear(in_features=config.hidden_size, out_features=config.patch_dim)
        
        # Generate positional embedding from utils
        self.positional_embedding = create_positional_encoding(
            image_shape=config.image_shape,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            is_video=config.is_video,
            num_frames=config.num_frames if config.is_video else None,
            theta=config.theta
        )
    
    def __call__(self, y: jax.Array | None = None):        
        # Get the initial sequence of patch embeddings from Vode 0
        x = self.vodes[0](jnp.empty(()))

        # Project patches to hidden dimension
        x = jax.vmap(self.patch_projection)(x)
        
        # Add positional embeddings
        x = x + self.positional_embedding

        # Apply patch projection vode
        x = self.vodes[1](x)

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            #TODO: apply residual connection here
            x = block(x) # Apply transformer block
            x = self.vodes[i+2](x) # Apply Vode
    
        # Project back to patch_dim
        x = jax.vmap(self.output_projection)(x)
        
        # Unpatchify back to image
        x = self.unpatchify(x, patch_size=self.config.patch_size, image_size=self.config.image_shape[1], channel_size=self.config.image_shape[0])
        
        # Apply sensory Vode
        x = self.vodes[-1](x)
        
        # Set target if provided
        if y is not None:
            self.vodes[-1].set("h", y)
        
        return self.vodes[-1].get("u")  # Return prediction
    
    # def __init__(self, config: TransformerConfig) -> None:
    #     super().__init__()
    #     self.config = px.static(config)
        
    #     # Initialize random key
    #     key = jax.random.PRNGKey(0)

    #     print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
    #     print(f"Model initialized with {self.config.hidden_size} hidden_size, {self.config.mlp_hidden_dim} mlp_hidden_dim")
    #     print(f"Using {'video' if self.config.is_video else 'image'} mode with shape {self.config.image_shape}")

    #     # Define Vodes for predictive coding
    #     # Top-level latent Vode
    #     self.vodes = [pxc.Vode(
    #         energy_fn=None,
    #         ruleset={
    #             pxc.STATUS.INIT: ("h, u <- u:to_init",),
    #             },
    #         tforms={
    #             "to_init": lambda n, k, v, rkg: jax.random.normal(
    #                 px.RKG(), (config.hidden_size,)
    #             ) * 0.01 if config.use_noise else jnp.zeros((config.hidden_size,))
    #         }
    #     )]
        
    #     # Create Vodes for each transformer block output except final one where we want different ruleset
    #     for _ in range(config.num_blocks):
    #         self.vodes.append(pxc.Vode(
    #             ruleset={
    #                 STATUS_FORWARD: ("h -> u",)}
    #         ))

    #     # Add final output Vode (sensory layer) - shape depends on whether we're handling video or images
    #     self.vodes.append(pxc.Vode())
        
    #     # Freeze the output Vode's hidden state
    #     self.vodes[-1].h.frozen = True  

    #     # DEBUG: Try FC blocks instead of transformer blocks to see if it works better for reconstruction
    #     self.fc_blocks = []
    #     for i in range(config.num_blocks):
    #         self.fc_blocks.append(
    #             pxnn.Linear(
    #                 in_features=config.hidden_size,
    #                 out_features=config.hidden_size
    #             )
    #         )

    #     # Add output projection layer to map from hidden_size back to output_dim
    #     self.output_projection = pxnn.Linear(in_features=config.hidden_size, out_features=32 * 32 * 3)
        
    
    # def __call__(self, y: jax.Array | None = None):        
    #     # Get the initial sequence of patch embeddings from Vode 0
    #     x = self.vodes[0](jnp.empty(()))

    #     # Process through FC blocks
    #     for i, block in enumerate(self.fc_blocks):
    #         x_after_block = block(x) 
    #         x = self.config.act_fn(x_after_block) 
    #         x = self.vodes[i+1](x) # Apply Vode

    #     # Apply output projection layer
    #     x = self.output_projection(x)
        
    #     # Reshape to match the expected image dimensions (channels, height, width)
    #     x = jnp.reshape(x, (3, 32, 32))
        
    #     # Apply sensory Vode
    #     x = self.vodes[-1](x)
        
    #     # Set target if provided
    #     if y is not None:
    #         self.vodes[-1].set("h", y)
        
    #     return self.vodes[-1].get("u")  # Return prediction
    

    def unpatchify(self, x, patch_size, image_size, channel_size):
        """
        Reconstructs a CIFAR-10 image from a sequence of patch embeddings.

        Args:
            x (array): Transformer output, shape (num_patches, patch_dim)
            patch_size (int): Size of each patch (e.g., 4 for 4x4 patches)
            image_size (int): Size of the image (e.g., 32 for 32x32 images)
            channel_size (int): Number of channels (e.g., 3 for RGB)

        Returns:
            image (array): Reconstructed image, shape (channel_size, image_size, image_size)
        """
        # Number of patches along each dimension (e.g., 32 // 4 = 8)
        num_patches_per_side = image_size // patch_size

        # Step 1: Reshape each patch embedding into (patch_size, patch_size, channel_size)
        # Input shape: (64, 48) -> (64, 4, 4, 3)
        x = einops.rearrange(
            x,
            'patches (p_h p_w c) -> patches p_h p_w c',
            p_h=patch_size,
            p_w=patch_size,
            c=channel_size
        )

        # Step 2: Rearrange patches into the full image grid
        # Shape: (64, 4, 4, 3) -> (3, 32, 32)
        image = einops.rearrange(
            x,
            '(h_num w_num) p_h p_w c -> c (h_num p_h) (w_num p_w)',
            h_num=num_patches_per_side,
            w_num=num_patches_per_side
        )

        return image


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: TransformerDecoder):
    return model(x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: TransformerDecoder):
    y_ = model(None)
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None, step=None):
    model.train()

    h_energy, w_energy, h_grad, w_grad = None, None, None, None

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

    # Top down sweep and setting target value
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference and learning steps
    # TODO: check how pxf.scan is used to wrap the T steps with JAX - this can speed up the loop
    for t in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (w_energy, y_), w_grad = learning_step(model=model)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])

    # After training, forward once more to get final activations
    # with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
    with pxu.step(model):
        x_hat_batch = forward(None, model=model) # Get final reconstruction

    # Calculate MSE loss between final reconstruction and original input
    x_hat_clipped = jnp.clip(x_hat_batch, 0.0, 1.0)
    x_clipped = jnp.clip(x, 0.0, 1.0)
    train_mse = jnp.mean((x_hat_clipped - x_clipped)**2)

    optim_h.clear()

    # Return energies, gradients, and the calculated training MSE
    return h_energy, w_energy, h_grad, w_grad, train_mse


def eval_pretext_metrics(dataloader, T_values, use_corruption, corrupt_ratio, *, model: TransformerDecoder, optim_h: pxu.Optim, data_range=1.0):
    """Evaluates pretext task performance (reconstruction/inpainting) on a dataloader."""
    model.eval() # Ensure model is in eval mode

    all_metrics = {
        'full_mse': [], 'masked_mse': [],
        'full_l1': [], 'masked_l1': [],
        'full_psnr': [], 'masked_psnr': []
    }

    total_batches = 0
    for x, _ in dataloader:
        total_batches += 1
        x_batch = jnp.array(x) # Ensure it's a JAX array

        # Use unmask_on_batch_enhanced to get reconstructions and the mask
        # We need the final reconstruction, which is the last element of the list
        # Note: This re-runs inference for every batch. Consider efficiency if needed.
        _, all_recons_list, mask = unmask_on_batch_enhanced(
            use_corruption=use_corruption,
            corrupt_ratio=corrupt_ratio,
            target_T_values=T_values, # Determines max_T
            x=x_batch,
            model=model,
            optim_h=optim_h
        )

        if not all_recons_list:
            print("Warning: No reconstructions returned from unmask_on_batch_enhanced.")
            continue

        # Get the final reconstruction (last step)
        x_hat_batch = all_recons_list[-1]

        # Ensure shapes match (unmask_on_batch_enhanced might return batched results)
        if x_hat_batch.shape[0] != x_batch.shape[0]:
             # If batch sizes don't match (e.g., due to internal repeat), take the first one
             # This assumes the relevant reconstruction is the first in the batch dimension returned.
             # Adjust if necessary based on how unmask_on_batch_enhanced handles batches.
             # It's safer if unmask_on_batch_enhanced consistently returns outputs matching the input batch size.
             print(f"Warning: Mismatch in batch size between input ({x_batch.shape[0]}) and reconstruction ({x_hat_batch.shape[0]}). Using first element.")
             # This logic might need refinement based on how your enhanced function handles batches
             # For now, let's assume we need to handle potential repeats
             if x_hat_batch.shape[0] % x_batch.shape[0] == 0:
                 repeats = x_hat_batch.shape[0] // x_batch.shape[0]
                 x_hat_batch = x_hat_batch[::repeats] # Take every Nth element if repeated
             else:
                  print("Cannot reconcile batch sizes, skipping batch metrics.")
                  continue # Skip if sizes are incompatible


        # Clip reconstructions to the expected data range for metrics
        # Assuming data is normalized to [0, 1] after potential [-1, 1] range
        x_hat_clipped = jnp.clip(x_hat_batch, 0.0, 1.0)
        x_clipped = jnp.clip(x_batch, 0.0, 1.0)

        # --- Calculate Full Image Metrics --- 
        full_mse = jnp.mean((x_hat_clipped - x_clipped)**2)
        full_l1 = jnp.mean(jnp.abs(x_hat_clipped - x_clipped))
        full_psnr = calculate_psnr(x_clipped, x_hat_clipped, data_range)

        all_metrics['full_mse'].append(full_mse)
        all_metrics['full_l1'].append(full_l1)
        all_metrics['full_psnr'].append(full_psnr)

        # --- Calculate Masked Image Metrics --- 
        # Ensure mask is broadcastable or has the same shape as x_batch
        mask = jnp.broadcast_to(mask, x_clipped.shape) # Ensure mask matches image shape
        num_masked_pixels = jnp.sum(mask)

        # Avoid division by zero if no pixels are masked
        if num_masked_pixels > 0:
            masked_diff_abs = jnp.abs(x_hat_clipped - x_clipped) * mask
            masked_diff_sq = ((x_hat_clipped - x_clipped) * mask)**2
            
            mean_masked_abs_diff = jnp.sum(masked_diff_abs) / num_masked_pixels
            masked_mse = jnp.sum(masked_diff_sq) / num_masked_pixels
            masked_l1 = mean_masked_abs_diff # Already calculated
            
            # For masked PSNR, calculate MSE on masked pixels first
            masked_psnr = calculate_psnr(x_clipped * mask, x_hat_clipped * mask, data_range)
        else: # If mask is all zeros
            masked_mse = 0.0
            masked_l1 = 0.0
            masked_psnr = 100.0 # Assign a high value indicating perfect match (or infinity)

        all_metrics['masked_mse'].append(masked_mse)
        all_metrics['masked_l1'].append(masked_l1)
        all_metrics['masked_psnr'].append(masked_psnr)

    # Average metrics over all batches
    avg_metrics = {key: jnp.mean(jnp.array(values)) for key, values in all_metrics.items() if values}
    
    print(f"Evaluated pretext metrics over {total_batches} batches.")
    return avg_metrics


def train(dl, T, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None):
    batch_w_energies = []
    batch_train_mses = []
    step = 0
    for x, y in dl:
        # Now returns train_mse as the 5th element
        h_energy, w_energy, h_grad, w_grad, train_mse = train_on_batch(
            T, jnp.array(x), model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step
        )
        
        if w_energy is not None:
            batch_w_energies.append(w_energy) 
        else:
            print(f"Warning: w_energy is None for batch {step}")
        
        if train_mse is not None:
             batch_train_mses.append(train_mse)
        else:
             print(f"Warning: train_mse is None for batch {step}")

        step += 1

    avg_train_w_energy = jnp.mean(jnp.array(batch_w_energies)) if batch_w_energies else 0.0
    avg_train_mse = jnp.mean(jnp.array(batch_train_mses)) if batch_train_mses else 0.0
    
    print(f"Epoch {epoch+1} Average Training w_energy: {avg_train_w_energy}")
    print(f"Epoch {epoch+1} Average Training MSE: {avg_train_mse}")
    
    # Return average w_energy, average mse, last gradients
    return avg_train_w_energy, avg_train_mse, h_grad, w_grad 


# @pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    # Init step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    # Inference steps (then we start to refine our guess)
    for t in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)
            print("h_energy:", h_energy, "at step t:", t)

        optim_h.step(model, h_grad["model"])
    
    optim_h.clear()

    # Final step (we make our final guess with our refined activations per layer)
    # with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
    # We do not clear the cache here as we want to keep the energy for logging
    with pxu.step(model, STATUS_FORWARD):
        x_hat = forward(None, model=model)

    loss = jnp.square(x_hat.flatten() - x.flatten()).mean()

    return loss, x_hat


def eval(dl, T, *, model: TransformerDecoder, optim_h: pxu.Optim):
    losses = []

    # TODO: think how having multiple batches modifies training dynamics. The lack of clear params for logging potentialy leads to issues.
    for x, y in dl:
        e, y_hat = unmask_on_batch(use_corruption=False, corrupt_ratio=0.0, target_T_values=T, x=jnp.array(x), model=model, optim_h=optim_h)
        losses.append(e)

    return jnp.mean(jnp.array(losses))


def unmask_on_batch(use_corruption: bool, corrupt_ratio: float, target_T_values: List[int], x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    """
    Runs inference on a batch (x), potentially corrupted, and returns the final loss
    and reconstructed outputs at specified T values.
    """
    model.eval()
    # TODO: in other scripts optim_h is not cleared and not initialised each time, only once. Check behavior.
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Define inference step with the regular energy function
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    max_T = max(target_T_values) if target_T_values else 0
    # Store reconstructions at target T values (step t corresponds to T=t+1)
    save_steps = {t - 1 for t in target_T_values if t > 0} 
    intermediate_recons = {}

    expected_bs = 1
    for vode in model.vodes:
        if vode.h._value is not None:
            expected_bs = vode.h._value.shape[0]
            break

    # Adjust batch size if needed
    if x.shape[0] != expected_bs:
        x_batch = jnp.repeat(x, expected_bs, axis=0)
    else:
        x_batch = x

    batch_size, channels, H, W = x_batch.shape
    assert model.config.image_shape == (channels, H, W), "Image shape mismatch"

    if use_corruption:
        # Create masked image
        x_c = x_batch.reshape((-1, 3, 32, 32)).copy()
        x_c = x_c.at[:, :, 16:].set(0)
        x_c = x_c.reshape((-1, 3, 32, 32))
        
        # Initialize the model with the masked input
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            forward(x_c, model=model)
    else:
        # Initialize the model with the input
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            forward(x_batch, model=model)

    # Inference iterations
    for t in range(max_T):
        if use_corruption:
            if t in save_steps:
                # with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                with pxu.step(model, STATUS_FORWARD):
                    intermediate_recons[t + 1] = forward(None, model=model)
                print(f"Saved intermediate reconstruction at T={t+1}")

            # Unfreeze sensory layer for inference
            model.vodes[-1].h.frozen = False

            # Run inference step
            # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            with pxu.step(model):
                h_energy, h_grad = inference_step(model=model)
            
            # Update states
            optim_h.step(model, h_grad["model"])

            model.vodes[-1].h.set(
                model.vodes[-1].h.reshape((-1, 3, 32, 32))
                .at[:, :, :16].set(
                    x_batch.reshape((-1, 3, 32, 32))
                    [:, :, :16]
                )
            )

        else:
            # Standard inference step
            # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            with pxu.step(model):
                h_energy, h_grad = inference_step(model=model)

            # Update states
            optim_h.step(model, h_grad["model"])

            if t in save_steps:
                #   with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                with pxu.step(model, STATUS_FORWARD):
                    intermediate_recons[t + 1] = forward(None, model=model)

    optim_h.clear()

    # Final forward pass to get reconstruction
    # with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
    # WARNING: Switched to this for vode grad and energy logging
    with pxu.step(model, STATUS_FORWARD):
        x_hat_batch = forward(None, model=model)

    loss = jnp.square(jnp.clip(x_hat_batch.reshape(-1), 0.0, 1.0) - x_batch.reshape(-1)).mean()

    # Refreeze sensory layer
    model.vodes[-1].h.frozen = True

    # WARNING: Commented out for vode grad and energy logging
    # # Reset model as the inference could mess up the model state if diverged
    # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
    #     forward(None, model=model)

    return loss, intermediate_recons


def calculate_psnr(img1, img2, data_range=1.0, epsilon=1e-8):
    """Calculates Peak Signal-to-Noise Ratio between two images."""
    mse = jnp.mean((img1 - img2) ** 2)
    # Handle potential division by zero or very small MSE
    safe_mse = jnp.maximum(mse, epsilon)
    psnr = 10 * jnp.log10((data_range ** 2) / safe_mse)
    # Clamp PSNR for the case where mse is exactly zero (perfect match)
    # Infinite PSNR isn't practical, often capped at a high value, e.g., 100 dB
    # JAX doesn't directly support inf, so we'll rely on the epsilon or cap if needed.
    # For now, epsilon handles the near-zero case. Perfect zero might still cause issues
    # depending on floating point precision, but epsilon helps.
    # A simpler alternative might be:
    # if mse == 0: return 100.0 # Or some large number
    # return 10 * jnp.log10((data_range ** 2) / mse)
    # However, JAX prefers numerical stability via epsilon.
    return psnr


def unmask_on_batch_enhanced(use_corruption: bool, corrupt_ratio: float, target_T_values: List[int], x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim) -> Tuple[jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
    """
    Enhanced version of unmask_on_batch with random masking, focused loss on masked regions,
    and detailed logging for debugging. Returns loss, list of reconstructions, and the mask used.
    """
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Define inference step with the regular energy function
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    max_T = max(target_T_values) if target_T_values else 0
    all_reconstructions = [] # Store all reconstructions

    expected_bs = 1
    for vode in model.vodes:
        if vode.h._value is not None:
            expected_bs = vode.h._value.shape[0]
            break

    # Adjust batch size if needed
    if x.shape[0] != expected_bs:
        x_batch = jnp.repeat(x, expected_bs, axis=0)
    else:
        x_batch = x

    batch_size, channels, H, W = x_batch.shape
    assert model.config.image_shape == (channels, H, W), "Image shape mismatch"

    mask = None 

    if use_corruption:
        # Determine masking type: random or lower half
        use_lower_half_mask = model.config.use_lower_half_mask  # Access config through model

        if use_lower_half_mask:
            # Create mask for lower half of the image
            mask = jnp.zeros((batch_size, 1, H, W))
            mask = mask.at[:, :, H//2:, :].set(1.0)
            x_c = jnp.where(mask == 1, 0.0, x_batch) # Set masked (lower half) to 0
        else:
            # Create masked image with random masking based on corrupt_ratio
            mask = jax.random.bernoulli(px.RKG(), p=corrupt_ratio, shape=(batch_size, 1, H, W))
            noise = jax.random.normal(px.RKG(), shape=x_batch.shape) * 0.1
            x_c = jnp.where(mask == 1, noise, x_batch)  # Masked regions get noise, unmasked get original
        
        # Initialize the model with the masked input
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            forward(x_c, model=model)
    else:
        # Initialize the model with the input
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            forward(x_batch, model=model)

    # Inference iterations
    # Wrap the range with tqdm for a progress bar
    for t in tqdm(range(max_T), desc="Inference Steps"):
        if use_corruption:
            # Always save reconstruction at each step
            with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                current_recon = forward(None, model=model)
                all_reconstructions.append(current_recon)
            # Unfreeze sensory layer for inference
            model.vodes[-1].h.frozen = False
            # Run inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (h_energy, y_), h_grad = inference_step(model=model)
            
            # # Zero out gradients for unmasked regions to focus updates on masked areas
            # sensory_h_grad = h_grad["model"].vodes[-1].h._value
            # # Reshape mask to match sensory layer output dimensions (batch_size, channels, H, W)
            # mask_broadcasted = mask[:, :, :, :]
            # # Repeat mask across the channel dimension to match sensory_h_grad shape
            # mask_broadcasted = jnp.repeat(mask_broadcasted, sensory_h_grad.shape[1], axis=1)
            # # Flatten mask and sensory gradients for element-wise operation
            # mask_flat = mask_broadcasted.reshape(sensory_h_grad.shape)
            # # Set gradients to 0 for masked regions (where mask is 1)
            # modified_sensory_h_grad = jnp.where(mask_flat == 1, 0.0, sensory_h_grad)
            # # Update the gradient value in h_grad
            # h_grad["model"].vodes[-1].h._value = modified_sensory_h_grad
            
            # Update states
            optim_h.step(model, h_grad["model"])

            # fix the value nodes of the sensory layer to the entries of the partial data point 
            # that we know are equal to the ones of the stored data point, leaving the rest free to be updated
            current_h = model.vodes[-1].h._value
            reset_output = current_h.reshape(x_batch.shape)
            reset_output = jnp.where(mask == 0, x_batch, reset_output)
            model.vodes[-1].h.set(reset_output)
        else:
            # Standard inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (h_energy, y_), h_grad = inference_step(model=model)
            # Update states
            optim_h.step(model, h_grad["model"])
            # Always save reconstruction at each step
            with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                current_recon = forward(None, model=model)
                all_reconstructions.append(current_recon)

    optim_h.clear()

    # Final forward pass to get reconstruction
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
         x_hat_batch = forward(None, model=model)

    # Compute loss, focusing on masked regions if corruption is used
    if use_corruption:
        if mask is None:
            # Handle error case, maybe assign a default loss or raise exception
            loss = -1.0 # Placeholder error value
        else:
            masked_loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch * mask, 0.0, 1.0) - x_batch * mask) / jnp.maximum(jnp.mean(mask), 1e-8)) # Normalize by mask ratio
            unmasked_loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch * (1 - mask), 0.0, 1.0) - x_batch * (1 - mask)) / jnp.maximum(1.0 - jnp.mean(mask), 1e-8)) # Normalize by inverse mask ratio
            loss = (masked_loss + unmasked_loss) / 2 # Average normalized losses
    else:
        loss = jnp.square(jnp.clip(x_hat_batch.reshape(-1), 0.0, 1.0) - x_batch.reshape(-1)).mean()
        if mask is None: # Ensure mask is assigned even if use_corruption was False
            mask = jnp.zeros_like(x_batch) 

    # Refreeze sensory layer
    model.vodes[-1].h.frozen = True

    # Reset model as the inference could mess up the model state if diverged. 
    # WARNING: too frequently resetting the model state causes training instability.
    # with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    # One final normal pass to get vode energies and gradients for logging
    with pxu.step(model):
        forward(None, model=model)
        
    # Return loss, reconstructions, and the mask
    return loss, all_reconstructions, mask


def create_config_by_dataset(dataset_name: str, latent_dim: int = 512, num_blocks: int = 6):
    """Create a TransformerConfig based on the dataset name."""
    # Define image_shape and other dataset-specific settings
    if dataset_name == "fashionmnist":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(1, 28, 28),
            hidden_size=256,
            num_heads=8,
            num_blocks=num_blocks,
            patch_size=4
        )
    elif dataset_name == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            hidden_size=256,
            num_heads=8,
            num_blocks=num_blocks,
            patch_size=4
        )
    elif dataset_name == "imagenet":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 224, 224),
            hidden_size=384,
            num_heads=8,
            num_blocks=num_blocks,
            patch_size=16
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    # Just an example of how to use the configuration system
    config = create_config_by_dataset(
        dataset_name="cifar10",
        latent_dim=512,
        num_blocks=6
    )
    
    # Create model with config
    model = TransformerDecoder(config)