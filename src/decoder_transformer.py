import multiprocessing
multiprocessing.set_start_method('spawn', force=True) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
            # print("h_energy:", h_energy, "at step t:", t)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (w_energy, y_), w_grad = learning_step(model=model)
            # print("w_energy:", w_energy, "at step t:", t)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])

    # After training, forward once more to get final activations
    # with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
    with pxu.step(model):
        forward(None, model=model)

    optim_h.clear()

    return h_energy, w_energy, h_grad, w_grad


def train(dl, T, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None):
    step = 0
    for x, y in dl:
        h_energy, w_energy, h_grad, w_grad = train_on_batch(T, jnp.array(x), model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step)
        print("w_energy:", w_energy, "at batch t:", step)
        step += 1

    return h_energy, w_energy, h_grad, w_grad


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


def unmask_on_batch_enhanced(use_corruption: bool, corrupt_ratio: float, target_T_values: List[int], x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    """
    Enhanced version of unmask_on_batch with random masking, focused loss on masked regions,
    and detailed logging for debugging. This function is an alternative to the original
    unmask_on_batch, allowing experimentation with improved unmasking logic.
    """
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Define inference step with the regular energy function
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    max_T = max(target_T_values) if target_T_values else 0
    # Store reconstructions at target T values (step t corresponds to T=t+1)
    # save_steps = {t - 1 for t in target_T_values if t > 0} 
    # intermediate_recons = {}
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

    if use_corruption:
        # Determine masking type: random or lower half
        use_lower_half_mask = model.config.use_lower_half_mask  # Access config through model

        if use_lower_half_mask:
            # Create mask for lower half of the image
            mask = jnp.zeros((batch_size, 1, H, W))
            # Set lower half to 1 (masked)
            mask = mask.at[:, :, H//2:, :].set(0.0)
            # Use zeros for lower half mask
            x_c = jnp.where(mask, 1.0, x_batch)  # Masked regions get zeros, unmasked get original
        else:
            # Create masked image with random masking based on corrupt_ratio
            mask = jax.random.bernoulli(px.RKG(), p=corrupt_ratio, shape=(batch_size, 1, H, W))
            # Use noise for random masking to avoid trivial solution
            noise = jax.random.normal(px.RKG(), shape=x_batch.shape) * 0.1
            x_c = jnp.where(mask, noise, x_batch)  # Masked regions get noise, unmasked get original
        
        # Log the masking ratio for debugging
        actual_mask_ratio = jnp.mean(mask)
        print(f"Applied random mask with target ratio {corrupt_ratio}, actual ratio: {actual_mask_ratio}")
        
        # Initialize the model with the masked input
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(x_c, model=model)
    else:
        # Initialize the model with the input
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(x_batch, model=model)

    # Inference iterations
    for t in range(max_T):
        if use_corruption:
            # Always save reconstruction at each step
            with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                current_recon = forward(None, model=model)
                all_reconstructions.append(current_recon)
            # print(f"Saved intermediate reconstruction at T={t+1}")

            # Unfreeze sensory layer for inference
            model.vodes[-1].h.frozen = False

            # Run inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (h_energy, y_), h_grad = inference_step(model=model)
            
            # Zero out gradients for unmasked regions to focus updates on masked areas
            sensory_h_grad = h_grad["model"].vodes[-1].h._value
            # Reshape mask to match sensory layer output dimensions (batch_size, channels, H, W)
            mask_broadcasted = mask[:, :, :, :]
            # Repeat mask across the channel dimension to match sensory_h_grad shape
            mask_broadcasted = jnp.repeat(mask_broadcasted, sensory_h_grad.shape[1], axis=1)
            # Flatten mask and sensory gradients for element-wise operation
            mask_flat = mask_broadcasted.reshape(sensory_h_grad.shape)
            # Set gradients to 0 for unmasked regions (where mask is 0)
            modified_sensory_h_grad = jnp.where(mask_flat == 0, 0.0, sensory_h_grad)
            # Update the gradient value in h_grad
            h_grad["model"].vodes[-1].h._value = modified_sensory_h_grad
            
            # Update states
            optim_h.step(model, h_grad["model"])

            # # Reset unmasked regions to original input to prevent degradation
            # reset_output = model.vodes[-1].h.reshape(x_batch.shape)
            # reset_output = jnp.where(mask == 0, x_batch, reset_output)
            # model.vodes[-1].h.set(reset_output)

            # # Log mean value of masked regions for debugging
            # masked_region_values = reset_output * mask
            # # Repeat mask across channels to match reset_output shape
            # mask_expanded = jnp.repeat(mask, reset_output.shape[1], axis=1)
            # mean_masked_value = jnp.mean(masked_region_values[mask_expanded > 0]) if jnp.any(mask_expanded > 0) else 0.0
            # print(f"Inference step {t+1}/{max_T}, Mean value in masked regions: {mean_masked_value}")

        else:
            # Standard inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (h_energy, y_), h_grad = inference_step(model=model)

            # Log energy for debugging
            print(f"Inference step {t+1}/{max_T}, Energy: {h_energy}")
            
            # Update states
            optim_h.step(model, h_grad["model"])

            # Always save reconstruction at each step
            with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                current_recon = forward(None, model=model)
                all_reconstructions.append(current_recon)
            # print(f"Saved intermediate reconstruction at T={t+1}")

    # Log energy for debugging - Note: h_energy is from the last step's inference_step call
    # print(f"Inference step {t+1}/{max_T}, Energy: {h_energy}")

    optim_h.clear()

    # Final forward pass to get reconstruction
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)

    # Compute loss, focusing on masked regions if corruption is used
    if use_corruption:
        masked_loss = jnp.square(jnp.clip(x_hat_batch * mask, 0.0, 1.0) - x_batch * mask).mean()
        unmasked_loss = jnp.square(jnp.clip(x_hat_batch * (1 - mask), 0.0, 1.0) - x_batch * (1 - mask)).mean()
        loss = masked_loss + unmasked_loss
        print(f"Final loss: {loss}, Masked loss: {masked_loss}, Unmasked loss: {unmasked_loss}")
    else:
        loss = jnp.square(jnp.clip(x_hat_batch.reshape(-1), 0.0, 1.0) - x_batch.reshape(-1)).mean()
        print(f"Final loss: {loss}")

    # Refreeze sensory layer
    model.vodes[-1].h.frozen = True

    # Reset model as the inference could mess up the model state if diverged. 
    # WARNING: too frequently resetting the model state causes training instability.
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(None, model=model)

    return loss, all_reconstructions


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