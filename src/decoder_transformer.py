# TODO: configs are all over the place, need to clean up
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
from pcx.core._parameter import get as param_get

import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

STATUS_FORWARD = "forward"

import jax.random as jrandom
key = jrandom.PRNGKey(42)

import jax.tree_util


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

    # Layer-specific inference LR scaling
    use_inference_lr_scaling: bool = False # Enable/disable scaling
    inference_lr_scale_lower: float = 1.0  # Multiplier for lower layers (Vodes < boundary)
    inference_lr_scale_upper: float = 1.0  # Multiplier for upper layers (Vodes >= boundary)
    inference_lr_scale_boundary: int = 3   # Index separating lower/upper 
    
    inference_clamp_alpha: float = 1.0     # Blending factor for soft clamping (1.0 = full clamp, <1.0 = blend)
    update_weights_during_unmasking: bool = False # New flag
    
    # Status init settings
    use_status_init_in_training: bool = True
    use_status_init_in_unmasking: bool = True

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
            # Apply residual connection
            x = x + block(x)
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


# class TransformerDecoder(pxc.EnergyModule):
#     def __init__(self, config: TransformerConfig) -> None:
#         super().__init__()
#         self.config = px.static(config)
        
#         # Initialize random key
#         key = jax.random.PRNGKey(0)

#         print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
#         print(f"Model initialized with {self.config.hidden_size} hidden_size, {self.config.mlp_hidden_dim} mlp_hidden_dim")
#         print(f"Using {'video' if self.config.is_video else 'image'} mode with shape {self.config.image_shape}")

#         # Define Vodes for predictive coding
#         # Top-level latent Vode
#         self.vodes = [pxc.Vode(
#             energy_fn=None,
#             ruleset={
#                 pxc.STATUS.INIT: ("h, u <- u:to_init",),
#                 },
#             tforms={
#                 "to_init": lambda n, k, v, rkg: jax.random.normal(
#                     px.RKG(), (config.hidden_size,)
#                 ) * 0.01 if config.use_noise else jnp.zeros((config.hidden_size,))
#             }
#         )]
        
#         # Create Vodes for each transformer block output except final one where we want different ruleset
#         for _ in range(config.num_blocks):
#             self.vodes.append(pxc.Vode(
#                 ruleset={
#                     STATUS_FORWARD: ("h -> u",)}
#             ))

#         # Add final output Vode (sensory layer) - shape depends on whether we're handling video or images
#         self.vodes.append(pxc.Vode())
        
#         # Freeze the output Vode's hidden state
#         self.vodes[-1].h.frozen = True  

#         # DEBUG: Try FC blocks instead of transformer blocks to see if it works better for reconstruction
#         self.fc_blocks = []
#         for i in range(config.num_blocks):
#             self.fc_blocks.append(
#                 pxnn.Linear(
#                     in_features=config.hidden_size,
#                     out_features=config.hidden_size
#                 )
#             )

#         # Add output projection layer to map from hidden_size back to output_dim
#         self.output_projection = pxnn.Linear(in_features=config.hidden_size, out_features=32 * 32 * 3)
        
    
#     def __call__(self, y: jax.Array | None = None):        
#         # Get the initial sequence of patch embeddings from Vode 0
#         x = self.vodes[0](jnp.empty(()))

#         # Process through FC blocks
#         for i, block in enumerate(self.fc_blocks):
#             x_after_block = block(x) 
#             x = self.config.act_fn(x_after_block) 
#             x = self.vodes[i+1](x) # Apply Vode

#         # Apply output projection layer
#         x = self.output_projection(x)
        
#         # Reshape to match the expected image dimensions (channels, height, width)
#         x = jnp.reshape(x, (3, 32, 32))
        
#         # Apply sensory Vode
#         x = self.vodes[-1](x)
        
#         # Set target if provided
#         if y is not None:
#             self.vodes[-1].set("h", y)
        
#         return self.vodes[-1].get("u")  # Return prediction
    

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
    initial_status_train = pxc.STATUS.INIT if model.config.use_status_init_in_training else None
    with pxu.step(model, initial_status_train, clear_params=pxc.VodeParam.Cache):
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
    with pxu.step(model, STATUS_FORWARD):
        x_hat_batch = forward(None, model=model)

    # Calculate MSE loss between final reconstruction and original input
    x_hat_clipped = jnp.clip(x_hat_batch, 0.0, 1.0)
    x_clipped = jnp.clip(x, 0.0, 1.0)
    train_mse = jnp.mean((x_hat_clipped - x_clipped)**2)

    optim_h.clear()

    # Return energies, gradients, and the calculated training MSE
    return h_energy, w_energy, h_grad, w_grad, train_mse


def eval_pretext_metrics(dataloader, T_values, use_corruption, corrupt_ratio, *, model: TransformerDecoder, optim_h: pxu.Optim, optim_w: Optional[pxu.Optim] = None, data_range=1.0):
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
        loss, all_recons_list, mask, _ = unmask_on_batch_enhanced(
            use_corruption=use_corruption,
            corrupt_ratio=corrupt_ratio,
            target_T_values=T_values, # Determines max_T
            x=x_batch,
            model=model,
            optim_h=optim_h,
            optim_w=optim_w
            # We don't need inference grads here, so log_inference_grads defaults to False
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


def scale_vode_grads(model_grads, lower_scale, upper_scale, boundary_idx):
    """Scales gradients for Vode hidden states based on layer index."""
    
    def map_fn(path, leaf):
        # Check if the leaf is a VodeParam corresponding to a hidden state gradient
        # The path for a vode gradient looks like: (..., 'vodes', index, 'h')
        is_vode_h_grad = (
            len(path) >= 3 and 
            path[-3] == 'vodes' and 
            isinstance(path[-2], jax.tree_util.SequenceKey) and # Check if it's a list index
            path[-1] == 'h'
        )
        
        if is_vode_h_grad:
            vode_idx = path[-2].idx
            scale = lower_scale if vode_idx < boundary_idx else upper_scale
            # Check if leaf is a parameter object with _value or just the array
            if hasattr(leaf, '_value') and leaf._value is not None:
                 # Return a new parameter object with scaled value
                 # Assuming leaf is a simple dataclass or object we can reconstruct
                 # This might need adjustment based on the exact type of VodeParam grad leaf
                 try:
                     # Attempt reconstruction assuming a simple structure
                     # This is a potential point of failure if VodeParam structure is complex
                     return type(leaf)(_value=leaf._value * scale, _frozen=leaf._frozen)
                 except:
                      print(f"Warning: Could not reconstruct VodeParam object for scaling at index {vode_idx}. Returning scaled value directly.")
                      return leaf._value * scale # Fallback: return only the scaled value
            elif isinstance(leaf, jax.Array): # If leaf is just the array
                 return leaf * scale
            else:
                return leaf # Return unchanged if not a Vode gradient or not an array/expected type
        else:
            return leaf # Return other parameters unchanged

    # Apply the mapping function to the gradient tree
    return jax.tree_util.tree_map_with_path(map_fn, model_grads)


def unmask_on_batch_enhanced(use_corruption: bool, corrupt_ratio: float, target_T_values: List[int], x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim, optim_w: Optional[pxu.Optim] = None, log_inference_grads: bool = False) -> Tuple[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Dict[int, Dict[str, float]]]:
    """
    Enhanced version of unmask_on_batch with random masking, focused loss on masked regions,
    and detailed logging for debugging. Returns loss, list of reconstructions, the mask used,
    and optionally, a log of gradient norms per inference step.
    """
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Define inference step with the regular energy function
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    max_T = max(target_T_values) if target_T_values else 0
    all_reconstructions = [] # Store all reconstructions
    inference_grad_norms_log = {} # Initialize log storage

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
    x_init = x_batch # Default initialization

    if use_corruption:
        print("use_corruption:", use_corruption)
        # Determine masking type: random or lower half
        use_lower_half_mask = model.config.use_lower_half_mask  # Access config through model

        if use_lower_half_mask:
            print("use_lower_half_mask:", use_lower_half_mask, "with ratio:", corrupt_ratio)
            # Calculate the number of rows to mask from the bottom based on corrupt_ratio
            # Clamp ratio just in case
            safe_corrupt_ratio = jax.lax.clamp(0.0, corrupt_ratio, 1.0) 
            num_rows_to_mask = jnp.floor(H * safe_corrupt_ratio).astype(jnp.int32)
            start_row = H - num_rows_to_mask
            
            # Create mask where bottom `num_rows_to_mask` rows are 1 (masked)
            mask = jnp.zeros((batch_size, 1, H, W))
            # Only apply mask if start_row is less than H (i.e., ratio > 0)
            if start_row < H: 
                mask = mask.at[:, :, start_row:, :].set(1.0)
                
            # Initialize corrupted image: masked region gets grey (0.5)
            placeholder_value = 0.5 # Use grey for masked areas
            x_c = jnp.where(mask == 1, placeholder_value, x_batch) 
        else:
            # Create masked image with random masking based on corrupt_ratio (mask=1 indicates masked region)
            mask = jax.random.bernoulli(px.RKG(), p=corrupt_ratio, shape=(batch_size, 1, H, W))
            noise = jax.random.normal(px.RKG(), shape=x_batch.shape) * 0.1 # Use small noise for masked regions
            x_c = jnp.where(mask == 1, noise, x_batch)  # Masked regions get noise, unmasked get original

        x_init = x_c # Initialize with the corrupted image

    # Initialize the model state based on x_init
    initial_status_unmask = pxc.STATUS.INIT if model.config.use_status_init_in_unmasking else None
    with pxu.step(model, initial_status_unmask, clear_params=pxc.VodeParam.Cache):
        forward(x_init, model=model)

    # Ensure sensory h is NOT frozen for inference
    model.vodes[-1].h.frozen = False

    # Inference iterations
    # Wrap the range with tqdm for a progress bar
    for t in tqdm(range(max_T), desc="Inference Steps"):
        # Always save reconstruction at each step (before clamping)
        with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            current_recon = forward(None, model=model)
            all_reconstructions.append(current_recon)

        # Run inference step (calculates energy and gradients based on current state)
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)

        # --- Log Gradients if requested ---
        if log_inference_grads:
            try:
                model_grads = h_grad['model']
                vodes_grads = model_grads.vodes
                current_step_norms = {}
                for i in range(len(vodes_grads)):
                    # Extract gradient tensor - accessing .h might give the VodeParam, need ._value
                    grad_param = vodes_grads[i].h
                    grad_values = grad_param._value if hasattr(grad_param, '_value') else None

                    if grad_values is not None:
                        # Calculate L2 norm using jnp.linalg.norm for stability
                        l2_norm = float(jnp.linalg.norm(grad_values.flatten()))
                        current_step_norms[f"vode_{i}_grad_norm"] = l2_norm
                    else:
                        # Handle case where gradient might be None (e.g., for frozen vodes, though we select non-frozen)
                        current_step_norms[f"vode_{i}_grad_norm"] = 0.0
                inference_grad_norms_log[t] = current_step_norms
            except Exception as e:
                print(f"Warning: Failed to log inference grads at step {t}: {e}")
                # Log empty dict or NaNs to avoid breaking structure if needed
                inference_grad_norms_log[t] = {f"vode_{i}_grad_norm": float('nan') for i in range(len(model.vodes))}
        # --- End Log Gradients ---

        # --- Apply Gradient Scaling if Enabled ---
        model_grads_to_apply = h_grad["model"]
        # Access config parameters directly through the model object
        if model.config.use_inference_lr_scaling:
            print(f"Applying inference LR scaling (Lower: x{model.config.inference_lr_scale_lower}, Upper: x{model.config.inference_lr_scale_upper}, Boundary: {model.config.inference_lr_scale_boundary})")
            model_grads_to_apply = scale_vode_grads(
                model_grads=model_grads_to_apply, 
                lower_scale=model.config.inference_lr_scale_lower,
                upper_scale=model.config.inference_lr_scale_upper,
                boundary_idx=model.config.inference_lr_scale_boundary
            )
        # --- End Gradient Scaling ---

        # Update ALL hidden states based on gradients (potentially scaled)
        optim_h.step(model, model_grads_to_apply)

        # --- Conditionally update weights ---
        if model.config.update_weights_during_unmasking and optim_w is not None:
            # Define learning_step similar to train_on_batch
            # WARNING: This selector pxu.M_hasnot(pxnn.LayerParam) calculates gradients for non-LayerParams.
            # If optim_w targets LayerParams, this might not be the intended gradient source.
            # Replicating train_on_batch logic as requested.
            learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)
            with pxu.step(model, clear_params=pxc.VodeParam.Cache): # Consistent with train_on_batch's weight update step
                (w_energy, y_w), w_grad = learning_step(model=model)
            # Ensure optim_w is initialized if it hasn't been (though typically it's initialized once)
            # For safety, one might check if optim_w.opt_state is initialized, but usually handled outside.
            optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0]) # Assuming x is the original batch for scaling
        # --- End conditional weight update ---

        # If corrupting, clamp the known (unmasked) pixels in the sensory layer *after* the update
        if use_corruption:
            current_h_sensory = model.vodes[-1].h._value
            mask_broadcast = jnp.broadcast_to(mask, x_batch.shape)
            alpha = model.config.inference_clamp_alpha # Get alpha from config
            
            # Calculate target state: GT for known, current state for unknown/masked
            target_h = jnp.where(mask_broadcast == 0, x_batch, current_h_sensory)
            
            # Blend the target state with the current state based on alpha
            blended_h = alpha * target_h + (1.0 - alpha) * current_h_sensory
            
            # Set the new blended state
            model.vodes[-1].h.set(blended_h)

    # No need for separate logic for non-corruption case inside the loop,
    # as clamping only happens if use_corruption is True.

    optim_h.clear()

    # Final forward pass to get reconstruction after all inference steps
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
    # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
         x_hat_batch = forward(None, model=model)

    # Compute loss, focusing on masked regions if corruption is used
    if use_corruption:
        if mask is None:
            # Handle error case, maybe assign a default loss or raise exception
            loss = -1.0 # Placeholder error value
            print("Error: Mask is None during loss calculation despite use_corruption=True")
        else:
            # Ensure mask is broadcast for loss calculation
            mask_broadcast_loss = jnp.broadcast_to(mask, x_hat_batch.shape)
            # Calculate MSE separately for masked and unmasked regions, then average
            # Normalize by the number of pixels in each region to avoid bias
            num_masked = jnp.sum(mask_broadcast_loss)
            num_unmasked = jnp.prod(jnp.array(x_hat_batch.shape)) - num_masked

            masked_error_sq = jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_batch) * mask_broadcast_loss
            unmasked_error_sq = jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_batch) * (1 - mask_broadcast_loss)

            masked_mse = jnp.sum(masked_error_sq) / jnp.maximum(num_masked, 1e-8)
            unmasked_mse = jnp.sum(unmasked_error_sq) / jnp.maximum(num_unmasked, 1e-8)

            loss = (masked_mse + unmasked_mse) / 2 # Average the two MSEs

            # Alternative: Calculate overall MSE (less informative about masked region performance)
            # loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_batch))
    else:
        # Standard MSE if no corruption was used
        loss = jnp.square(jnp.clip(x_hat_batch.reshape(-1), 0.0, 1.0) - x_batch.reshape(-1)).mean()
        if mask is None: # Ensure mask is assigned even if use_corruption was False
            mask = jnp.zeros_like(x_batch) # Assign a zero mask if needed for return type consistency

    # Refreeze sensory layer after inference is complete
    model.vodes[-1].h.frozen = True

    # Reset model state? Maybe not necessary here, let's see.
    # The subsequent training epoch will re-initialize anyway.
    # Keep the final pass for potential logging if needed elsewhere.
    with pxu.step(model):
        forward(None, model=model)

    # Return loss, reconstructions, mask, and the grad norm log
    return loss, all_reconstructions, mask, inference_grad_norms_log


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