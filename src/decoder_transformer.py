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
from jax.typing import DTypeLike
from einops import rearrange
from pcx.core._parameter import get as param_get
from pcx.predictive_coding import regularized_plus_se_energy, se_energy

import matplotlib.pyplot as plt
from datetime import datetime
import time
import math
import functools

STATUS_FORWARD = "forward"

import jax.random as jrandom
key = jrandom.PRNGKey(42)

import jax.tree_util

from src.mmcr_loss import calculate_mmcr_loss_for_vode
from src.utils import create_positional_encoding, apply_exponential_layer_scaling, normalize_vode_h_gradients, calculate_psnr
from pcx.nn._stateful import BatchNorm

# Import moved to avoid circular dependency
# from src.utils import create_positional_encoding

# Import TransformerConfig from a separate config file to avoid circular imports
try:
    from src.config import TransformerConfig
except ImportError as e:
    print(f"Error importing TransformerConfig: {e}")
    raise

# Import utility functions after class definitions to avoid circular imports
try:
    from src.utils import create_positional_encoding, apply_exponential_layer_scaling, normalize_vode_h_gradients, calculate_psnr
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    raise


class Projector(px.Module):
    """A projector network with Linear -> BatchNorm -> ReLU -> Linear."""
    linear1: pxnn.Linear
    # bn: BatchNorm
    linear2: pxnn.Linear

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rkg: px.RandomKeyGenerator):
        super().__init__()
        self.linear1 = pxnn.Linear(in_dim, hidden_dim, bias=False, rkg=rkg)
        # self.bn = BatchNorm(input_size=hidden_dim, axis_name=None)
        self.linear2 = pxnn.Linear(hidden_dim, out_dim, rkg=rkg)

    def __call__(self, x: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        # Linear layers are stateless and operate per-example, so we vmap them.
        x = jax.vmap(self.linear1)(x)
        # BatchNorm is stateful and operates on the whole batch.
        # x = self.bn(x, key=key)
        x = jax.nn.relu(x)
        # The second linear layer.
        x = jax.vmap(self.linear2)(x)
        return x


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = px.static(config)
        
        # Initialize random key
        key = jax.random.PRNGKey(0)

        print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
        print(f"Model initialized with {self.config.hidden_size} hidden_size, {self.config.mlp_hidden_dim} mlp_hidden_dim")
        print(f"Using {'video' if self.config.is_video else 'image'} mode with shape {self.config.image_shape}")
        print(f"Regularization: IntL1={self.config.intermediate_l1_coeff}, IntL2={self.config.intermediate_l2_coeff}")

        # Define Vodes for predictive coding
        # Top-level latent Vode
        self.vodes = [pxc.Vode(
            energy_fn=None, # Vode 0 does not have local energy term for regularization
            ruleset={
                pxc.STATUS.INIT: ("h, u <- u:to_init",)
                },
            tforms={
                "to_init": lambda n, k, v, rkg: jax.random.normal(
                    px.RKG(), (config.num_patches, config.patch_dim)
                ) * 0.01 if config.use_noise else jnp.zeros((config.num_patches, config.patch_dim))
            }
        )]

        # Intermediate Vodes energy function (Patch Projection Vode and Transformer Block Vodes)
        # If intermediate_l1_coeff and intermediate_l2_coeff are 0, 
        # regularized_plus_se_energy will effectively just compute its SE term.
        intermediate_vodes_energy_fn = functools.partial(
            regularized_plus_se_energy,
            l1_coeff=self.config.intermediate_l1_coeff,
            l2_coeff=self.config.intermediate_l2_coeff
        )

        # Add a Vode for patch projection (Vode 1)
        self.vodes.append(pxc.Vode(
            energy_fn=intermediate_vodes_energy_fn, # Use the prepared partial function
                ruleset={ 
                    STATUS_FORWARD: ("h -> u",)}
            ))
        
        # Create Vodes for each transformer block output (Vodes 2 to num_blocks + 1)
        for _ in range(config.num_blocks):
            self.vodes.append(pxc.Vode(
                energy_fn=intermediate_vodes_energy_fn, # Use the same prepared partial function
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
        
        # Add projector heads for MMCR loss
        self.projection_heads = []
        if self.config.use_mmcr_loss:
            # Add a dynamic, non-static scale factor for MMCR loss
            self.mmcr_loss_scale_factor = jnp.array(self.config.mmcr_loss_scale_factor, dtype=jnp.float32)
            if self.config.mmcr_vode_indices is None:
                raise ValueError("mmcr_vode_indices must be specified when use_mmcr_loss is True")

            for i in range(config.num_blocks + 2):
                if i in self.config.mmcr_vode_indices:
                    # Determine input dimension for the projector
                    if i == 0:
                        in_dim = config.patch_dim
                    else:
                        in_dim = config.hidden_size
                    
                    projector = Projector(
                        in_dim=in_dim,
                        hidden_dim=self.config.mmcr_projector_hidden_dim,
                        out_dim=self.config.mmcr_projector_dim,
                        rkg=px.RKG
                    )
                    self.projection_heads.append(projector)
                else:
                    self.projection_heads.append(None)
        
        # Generate positional embedding from utils
        self.positional_embedding = create_positional_encoding(
            image_shape=config.image_shape,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            is_video=config.is_video,
            num_frames=config.num_frames if config.is_video else None,
            theta=config.theta
        )

        # LayerNorms for Vode states, if enabled
        self.vode_output_layernorms = []
        if self.config.use_vode_state_layernorm:
            # LayerNorm after patch projection Vode (Vode 1)
            self.vode_output_layernorms.append(pxnn.LayerNorm(shape=(config.hidden_size,)))
            # LayerNorms after each Transformer block Vode
            for _ in range(config.num_blocks):
                self.vode_output_layernorms.append(pxnn.LayerNorm(shape=(config.hidden_size,)))
    

    def __call__(self, y: jax.Array | None = None):        
        # Get the initial sequence of patch embeddings from Vode 0
        x = self.vodes[0](jnp.empty(()))

        # Project patches to hidden dimension
        x = jax.vmap(self.patch_projection)(x)
        
        # Add positional embeddings
        x = x + self.positional_embedding

        # Apply patch projection vode
        x = self.vodes[1](x)
        if self.config.use_vode_state_layernorm and self.vode_output_layernorms:
            x = jax.vmap(self.vode_output_layernorms[0])(x)

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Apply residual connection
            x = x + block(x)
            x = self.vodes[i+2](x) # Apply Vode
            if self.config.use_vode_state_layernorm and len(self.vode_output_layernorms) > (i + 1):
                x = jax.vmap(self.vode_output_layernorms[i+1])(x)
    
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


def batch_energy(*, model: TransformerDecoder):
    y_ = forward(None, model=model)
    e = energy(model=model)[0]
    return jnp.mean(e), y_


@pxf.jit(static_argnums=0, static_argnames=("b_orig", "n_views"))
def train_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None, step=None, b_orig: int = -1, n_views: int = -1):
    model.train()

    # Perform checks outside JIT-ted functions
    if model.config.use_mmcr_loss and (b_orig == -1 or n_views == -1):
        raise ValueError("b_orig and n_views must be provided when use_mmcr_loss is True")

    # Define the total energy function, which now includes MMCR
    def total_energy_fn(model):

        # The model's forward pass populates all h and u states
        # Base reconstruction energy (sum of all Vode energies)
        reconstruction_energy, _ = batch_energy(model=model)

        # MMCR Loss Calculation
        mmcr_energy = jnp.array(0.0)
        if model.config.use_mmcr_loss:
            # The check for b_orig and n_views has been moved outside this function
            for i, projector in enumerate(model.projection_heads):
                if projector is not None:
                    # Get the hidden state `h` from the corresponding Vode
                    h_i = model.vodes[i].get("h")
                    
                    # Handle different possible shapes of h_i
                    if h_i.ndim == 3:  # (B, num_patches, hidden_dim)
                        # Average pooling over patches
                        h_i_pooled = jnp.mean(h_i, axis=1)  # Shape: (B, hidden_dim)
                    elif h_i.ndim == 2:  # (B, hidden_dim)
                        h_i_pooled = h_i
                    else:
                        raise ValueError(f"Unexpected shape for h_i: {h_i.shape}")
                    
                    # Project to get 'z'
                    z_i = projector(h_i_pooled)

                    # Calculate MMCR loss for this layer
                    mmcr_loss_i = calculate_mmcr_loss_for_vode(
                        z_i,
                        b_orig=b_orig,
                        n_views=n_views,
                        mmcr_lambda=model.config.mmcr_lambda
                    )
                    mmcr_energy += mmcr_loss_i
        
        # Use the dynamic scale factor from the model object
        scaled_mmcr_energy = mmcr_energy * model.mmcr_loss_scale_factor
        
        total_energy = reconstruction_energy + scaled_mmcr_energy
        
        # We need an aux output to see the components
        return total_energy, (reconstruction_energy, scaled_mmcr_energy)

    # Define functions to get energies and gradients for hidden states (h) and weights (w)
    def get_energies_and_grads_h(model_for_grad):
        grad_fn = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(total_energy_fn)
        (total_and_aux, grads) = grad_fn(model=model_for_grad)
        return total_and_aux, grads


    def get_energies_and_grads_w(model_for_grad):
        grad_fn = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(total_energy_fn)
        (total_and_aux, grads) = grad_fn(model=model_for_grad)
        return total_and_aux, grads


    # Initial forward pass and optimizer init
    initial_status_train = pxc.STATUS.INIT if model.config.use_status_init_in_training else None
    with pxu.step(model, initial_status_train, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    # Ensure model status and individual Vode statuses are not INIT before entering scan
    if initial_status_train == pxc.STATUS.INIT:
        model.status = None
        for vode in model.vodes:
            vode.status = None

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    dummy_carry_init = None

    def inference_only_iteration_for_scan(iter_idx_ignored, carry_dummy, *, model_closure, optim_h_closure):
        with pxu.step(model_closure, clear_params=pxc.VodeParam.Cache):
            (energies, h_grad) = get_energies_and_grads_h(model_closure)
            total_h, _ = energies
        
        # Apply scaling if enabled, directly modifying model_grads_to_apply
        model_grads_to_apply = h_grad["model"] 
        if model_closure.config.use_inference_lr_scaling:
            model_grads_to_apply = apply_exponential_layer_scaling(
                model_grads=model_grads_to_apply,
                config=model_closure.config
            )
        
        # New: Apply Vode h-gradient normalization if enabled
        if model_closure.config.use_vode_grad_norm:
            model_grads_to_apply = normalize_vode_h_gradients(
                model_grads=model_grads_to_apply, # Pass the potentially already scaled grads
                config=model_closure.config
            )

        # Update hidden states using the (potentially scaled and/or normalized) gradients
        optim_h_closure.step(model_closure, model_grads_to_apply)

        # We need to return the full h_grad structure but with potentially scaled model grads
        final_grad_structure = h_grad.copy() 
        final_grad_structure["model"] = model_grads_to_apply # Directly use the applied grads

        return carry_dummy, (total_h, final_grad_structure) 

    _, (h_energies_scanned, h_grads_potentially_scaled_scanned) = pxf.scan(inference_only_iteration_for_scan, xs=jnp.arange(T)
    )(dummy_carry_init, model_closure=model, optim_h_closure=optim_h)

    final_h_energy = h_energies_scanned[-1]

    final_h_grad = jax.tree_util.tree_map(lambda leaf: leaf[-1], h_grads_potentially_scaled_scanned)
    
    # Single learning step after all T inference steps
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (energies, final_w_grad) = get_energies_and_grads_w(model)
        final_w_energy, ((final_recons_energy, final_mmcr_energy),) = energies
    optim_w.step(model, final_w_grad["model"], scale_by=1.0/x.shape[0])
    
    # Common post-loop steps (MSE calculation, etc.)
    with pxu.step(model, STATUS_FORWARD):
        x_hat_batch = forward(None, model=model)

    # Calculate MSE loss between final reconstruction and original input
    x_hat_clipped = jnp.clip(x_hat_batch, 0.0, 1.0)
    x_clipped = jnp.clip(x, 0.0, 1.0)
    train_mse = jnp.mean((x_hat_clipped - x_clipped)**2)

    optim_h.clear()

    # Return energies, gradients, and the calculated training MSE
    return final_h_energy, final_w_energy, final_recons_energy, final_mmcr_energy, final_h_grad, final_w_grad, train_mse


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
    batch_recons_energies = []
    batch_mmcr_energies = []
    batch_train_mses = []
    step = 0
    last_postfix_update_time = time.time() # For throttling postfix updates
    
    # Initialize gradients to None in case the dataloader is empty
    h_grad, w_grad = None, None

    pbar = tqdm(dl, desc=f"Epoch {epoch+1 if epoch is not None else 'N/A'}", leave=True, mininterval=1.0)
    
    for x, y in pbar:
        # x is now a batch of multi-view augmentations, shape (b_orig, n_views, C, H, W)
        b_orig, n_views, c, h, w = x.shape
        # Reshape for the model: (b_orig * n_views, C, H, W)
        x_reshaped = x.reshape((b_orig * n_views, c, h, w))
        
        # The new train_on_batch returns more energy components
        h_energy, w_energy, recons_energy, mmcr_energy, h_grad, w_grad, train_mse = train_on_batch(
            T, jnp.array(x_reshaped), model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step,
            b_orig=b_orig, n_views=n_views
        )
        
        if w_energy is not None:
            batch_w_energies.append(w_energy)
        if recons_energy is not None:
            batch_recons_energies.append(recons_energy)
        if mmcr_energy is not None:
            batch_mmcr_energies.append(mmcr_energy)
        if train_mse is not None:
             batch_train_mses.append(train_mse)

        # Update progress bar postfix, throttled to once per second
        current_time = time.time()
        if current_time - last_postfix_update_time >= 1.0:
            metrics_to_display = {}
            if batch_train_mses:
                mse_to_avg = batch_train_mses[-100:]
                avg_recent_mse = jnp.mean(jnp.array(mse_to_avg))
                metrics_to_display['Avg_MSE_100'] = f'{avg_recent_mse:.4f}'
            if batch_recons_energies:
                energy_to_avg = batch_recons_energies[-100:]
                avg_recent_recons = jnp.mean(jnp.array(energy_to_avg))
                metrics_to_display['Avg_Recons_100'] = f'{avg_recent_recons:.2f}'
            if batch_mmcr_energies:
                energy_to_avg = batch_mmcr_energies[-100:]
                avg_recent_mmcr = jnp.mean(jnp.array(energy_to_avg))
                metrics_to_display['Avg_MMCR_100'] = f'{avg_recent_mmcr:.2f}'
            
            if metrics_to_display:
                pbar.set_postfix(metrics_to_display)
                last_postfix_update_time = current_time

        step += 1

    avg_train_w_energy = jnp.mean(jnp.array(batch_w_energies)) if batch_w_energies else 0.0
    avg_recons_energy = jnp.mean(jnp.array(batch_recons_energies)) if batch_recons_energies else 0.0
    avg_mmcr_energy = jnp.mean(jnp.array(batch_mmcr_energies)) if batch_mmcr_energies else 0.0
    avg_train_mse = jnp.mean(jnp.array(batch_train_mses)) if batch_train_mses else 0.0
    
    print(f"Epoch {epoch+1} Avg Total Energy: {avg_train_w_energy:.4f}, Avg Recons Energy: {avg_recons_energy:.4f}, Avg MMCR Energy: {avg_mmcr_energy:.4f}, Avg Train MSE: {avg_train_mse:.4f}")
    
    # Return all relevant metrics
    return avg_train_w_energy, avg_recons_energy, avg_mmcr_energy, avg_train_mse, h_grad, w_grad


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

    # Control sensory layer frozen status based on corruption
    if use_corruption:
        # For corruption/inpainting, sensory h needs to be dynamic for clamping and updates.
        model.vodes[-1].h.frozen = False
    else:
        # For standard reconstruction visualization (matching training target setup),
        # sensory h should be fixed to the input image.
        # It was already set by forward(x_init, model=model).
        model.vodes[-1].h.frozen = True

    # Inference iterations
    # Wrap the range with tqdm for a progress bar
    for t in tqdm(range(max_T), desc="Inference Steps"):
        # Always save reconstruction at each step (before clamping)
        with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
            current_recon = forward(None, model=model)
            all_reconstructions.append(current_recon)

        # Run inference step (calculates energy and gradients based on current state)
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)

        # Apply Gradient Scaling if Enabled
        model_grads_to_apply = h_grad["model"]
        scaled_grads_used_inference = False
        if model.config.use_inference_lr_scaling:
            model_grads_to_apply = apply_exponential_layer_scaling(
                model_grads=model_grads_to_apply,
                config=model.config
            )
            scaled_grads_used_inference = True

        # --- Log Gradients if requested ---
        if log_inference_grads:
            try:
                # Use the grads that will be used for the update step for norm calculation
                grads_for_norm_calc = model_grads_to_apply # These are potentially scaled
                vodes_grads = grads_for_norm_calc.vodes # Access vodes within the potentially scaled structure
                current_step_norms = {}
                for i in range(len(vodes_grads)):
                    grad_param = vodes_grads[i].h
                    # Handle cases where the value might be accessed differently
                    # depending on whether scaling returned the object or just the value
                    if hasattr(grad_param, '_value'):
                        grad_values = grad_param._value
                    elif isinstance(grad_param, jax.Array): # Check if it's just the array (fallback from scaling)
                        grad_values = grad_param
                    else:
                        grad_values = None # Or handle other potential structures

                    if grad_values is not None:
                        l2_norm = float(jnp.linalg.norm(grad_values.flatten()))
                        current_step_norms[f"vode_{i}_grad_norm"] = l2_norm
                    else:
                        current_step_norms[f"vode_{i}_grad_norm"] = 0.0 # Or NaN
                inference_grad_norms_log[t] = current_step_norms
            except Exception as e:
                print(f"Warning: Failed to log inference grads at step {t}: {e}")
                inference_grad_norms_log[t] = {f"vode_{i}_grad_norm": float('nan') for i in range(len(model.vodes))}
        # --- End Log Gradients ---

        # Update ALL hidden states using the (potentially scaled) gradients
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


if __name__ == "__main__":    
    pass