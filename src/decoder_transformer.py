import multiprocessing
multiprocessing.set_start_method('spawn', force=True) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from typing import Callable, List, Optional, Dict, Any, Tuple
from functools import partial
from contextlib import contextmanager 
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
from flax import nnx
from jax.typing import DTypeLike
from einops import rearrange
from pcx.core._parameter import get as param_get  # Import get function for parameter access

STATUS_FORWARD = "forward"
STATUS_REFINE = "refine"

import jax.random as jrandom
key = jrandom.PRNGKey(42)


class ModelDebugger:
    """Utility class for debugging model training and inference."""
    
    def __init__(self, enable_logging=True, log_dir="../debug_logs"):
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.energy_history = []
        self.gradient_norm_history = []
        self.activation_stats = {}
        
        if enable_logging:
            os.makedirs(log_dir, exist_ok=True)
    
    def log_energy(self, step, h_energy, w_energy):
        """Log energy values during training."""
        if not self.enable_logging:
            return
            
        self.energy_history.append({
            'step': step,
            'h_energy': float(h_energy) if h_energy is not None else None,
            'w_energy': float(w_energy) if w_energy is not None else None
        })
        
        # Save periodically to avoid memory issues
        if len(self.energy_history) % 100 == 0:
            self._save_energy_log()
    
    def log_gradient_norms(self, step, h_grad, w_grad):
        """Log gradient norms to detect exploding/vanishing gradients."""
        if not self.enable_logging or h_grad is None or w_grad is None:
            return
            
        # Extract gradients from the model parameters
        h_grad_flat = self._flatten_gradients(h_grad.get("model", {}))
        w_grad_flat = self._flatten_gradients(w_grad.get("model", {}))
        
        # Compute norms
        h_grad_norm = jnp.linalg.norm(h_grad_flat) if h_grad_flat is not None else None
        w_grad_norm = jnp.linalg.norm(w_grad_flat) if w_grad_flat is not None else None
        
        self.gradient_norm_history.append({
            'step': step,
            'h_grad_norm': float(h_grad_norm) if h_grad_norm is not None else None,
            'w_grad_norm': float(w_grad_norm) if w_grad_norm is not None else None
        })
        
        # Save periodically
        if len(self.gradient_norm_history) % 100 == 0:
            self._save_gradient_log()
    
    def log_activation_stats(self, name, activation):
        """Log statistics about activations at different layers."""
        if not self.enable_logging:
            return
            
        try:
            # Use numpy to avoid JAX tracer issues - carefully handle JAX arrays
            if hasattr(activation, 'numpy'):
                # Direct numpy conversion for PyTorch tensors
                activation_np = activation.numpy()
            elif hasattr(activation, 'shape') and hasattr(jnp, 'asarray'):
                # For JAX arrays, must first check if concrete (not a tracer)
                # Use np.asarray for already-concrete arrays
                if not hasattr(activation, '_trace'):
                    try:
                        activation_np = np.asarray(activation)
                    except Exception as e1:
                        print(f"Error converting activation {name} to numpy: {e1}")
                        return
                else:
                    # Skip if it's a JAX tracer (inside JIT context)
                    print(f"Warning: Skipping logging for {name} because it's a JAX tracer")
                    return
            else:
                # Skip if we can't safely convert to numpy
                print(f"Warning: Skipping logging for {name} (can't convert to numpy)")
                return
            
            # Calculate statistics safely
            stats = {}
            try:
                stats['min'] = float(np.min(activation_np))
                stats['max'] = float(np.max(activation_np))
                stats['mean'] = float(np.mean(activation_np))
                stats['std'] = float(np.std(activation_np))
                stats['abs_mean'] = float(np.mean(np.abs(activation_np)))
                stats['zero_frac'] = float(np.mean(np.equal(activation_np, 0)))
            except Exception as e2:
                print(f"Error calculating statistics for {name}: {e2}")
                return
            
            if name not in self.activation_stats:
                self.activation_stats[name] = []
            
            self.activation_stats[name].append(stats)
            
            # Save periodically
            if len(self.activation_stats[name]) % 20 == 0:
                self._save_activation_log(name)
        except Exception as e:
            print(f"Error logging activation stats for {name}: {e}")
            # Continue without failing if there's an error
    
    def visualize_layer_output(self, model, name, output, epoch=None):
        """Visualize the output of specific layers."""
        if not self.enable_logging:
            return
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # For image-like outputs only
        if len(output.shape) < 3:
            return
            
        # Create directory for visualizations
        viz_dir = os.path.join(self.log_dir, "layer_viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        
        # Save visualization
        plt.figure(figsize=(8, 8))
        
        # Reshape and normalize for visualization
        if len(output.shape) == 3:  # [B, L, D]
            # Reshape to square if possible
            side = int(np.sqrt(output.shape[1]))
            if side**2 == output.shape[1]:
                viz = output[0].reshape(side, side, -1)
                # Take average across feature dimension
                viz = jnp.mean(viz, axis=-1)
                plt.imshow(viz, cmap='viridis')
            else:
                # Just visualize the first few dimensions as a heatmap
                viz = output[0, :100, :100] if output.shape[1] > 100 and output.shape[2] > 100 else output[0]
                plt.imshow(viz, cmap='viridis', aspect='auto')
        elif len(output.shape) == 4:  # Image-like
            if output.shape[1] == 3:  # [B, C, H, W] format
                viz = jnp.transpose(output[0], (1, 2, 0))
                plt.imshow(jnp.clip(viz, 0, 1))
            else:
                # Take first channel
                viz = output[0, 0]
                plt.imshow(viz, cmap='viridis')
        
        plt.title(f"Layer: {name}")
        plt.colorbar()
        plt.savefig(f"{viz_dir}/{name}_{timestamp}{epoch_str}.png")
        plt.close()
    
    def debug_unpatchify(self, model, patched_output, epoch=None):
        """Debug the unpatchify process by visualizing before and after."""
        if not self.enable_logging:
            return
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # Create directory for unpatchify debugging
        viz_dir = os.path.join(self.log_dir, "unpatchify_debug")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        
        # Visualize patched representation
        plt.figure(figsize=(12, 10))
        
        # Plot heatmaps of patches
        plt.subplot(2, 2, 1)
        # Take first 16 patches for visualization
        num_patches = min(16, patched_output.shape[1])
        patch_dim = patched_output.shape[2]
        
        # Reshape to visualize patch content
        patch_viz = patched_output[0, :num_patches, :].reshape(num_patches, -1)
        plt.imshow(patch_viz, cmap='viridis', aspect='auto')
        plt.title("Patch Representation (first 16 patches)")
        plt.colorbar()
        
        # Plot histogram of values in patches
        plt.subplot(2, 2, 2)
        plt.hist(patched_output[0].flatten(), bins=50, alpha=0.7)
        plt.title("Distribution of Values in Patches")
        
        # Try to unpatchify and visualize
        try:
            img_shape = model.config.image_shape
            unpatchified = model._unpatchify(patched_output, batch_size=patched_output.shape[0])
            
            plt.subplot(2, 2, 3)
            if img_shape[0] == 1:  # Grayscale
                plt.imshow(jnp.clip(jnp.squeeze(unpatchified[0]), 0, 1), cmap='gray')
            else:  # RGB
                plt.imshow(jnp.clip(jnp.transpose(unpatchified[0], (1, 2, 0)), 0, 1))
            plt.title("Unpatchified Image")
            
            plt.subplot(2, 2, 4)
            plt.hist(unpatchified[0].flatten(), bins=50, alpha=0.7)
            plt.title("Distribution of Values in Unpatchified Image")
            
        except Exception as e:
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, f"Error in unpatchify: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/unpatchify_debug_{timestamp}{epoch_str}.png")
        plt.close()
    
    def _flatten_gradients(self, grad_dict):
        """Flatten gradient dictionary into a single array for norm calculation."""
        try:
            flat_arrays = []
            
            # If grad_dict is not a dictionary but has a get() method (like the model object)
            if hasattr(grad_dict, 'get') and not isinstance(grad_dict, dict):
                return self._flatten_gradients(grad_dict.get("model", {}))
            
            # Check if we have a model attribute and handle that case
            if hasattr(grad_dict, 'model') and not isinstance(grad_dict, dict):
                return self._flatten_gradients(grad_dict.model)
            
            # Handle dictionary case
            if isinstance(grad_dict, dict):
                for k, v in grad_dict.items():
                    if isinstance(v, dict):
                        # Recursively flatten nested dictionaries
                        nested_flat = self._flatten_gradients(v)
                        if nested_flat is not None:
                            flat_arrays.append(nested_flat)
                    elif isinstance(v, jnp.ndarray):
                        flat_arrays.append(v.flatten())
            
            if flat_arrays:
                return jnp.concatenate(flat_arrays)
            return None
        except Exception as e:
            print(f"Error flattening gradients: {e}")
            return None
    
    def _save_energy_log(self):
        """Save energy history to file."""
        if not self.energy_history:
            return
            
        import json
        with open(os.path.join(self.log_dir, 'energy_history.json'), 'w') as f:
            json.dump(self.energy_history, f)
    
    def _save_gradient_log(self):
        """Save gradient norm history to file."""
        if not self.gradient_norm_history:
            return
            
        import json
        with open(os.path.join(self.log_dir, 'gradient_norm_history.json'), 'w') as f:
            json.dump(self.gradient_norm_history, f)
    
    def _save_activation_log(self, name):
        """Save activation statistics for a specific layer."""
        if name not in self.activation_stats or not self.activation_stats[name]:
            return
            
        import json
        with open(os.path.join(self.log_dir, f'activation_{name}.json'), 'w') as f:
            json.dump(self.activation_stats[name], f)
    
    def save_all_logs(self):
        """Save all logs to files."""
        if not self.enable_logging:
            return
            
        self._save_energy_log()
        self._save_gradient_log()
        
        for name in self.activation_stats:
            self._save_activation_log(name)

# Create a global debugger instance
model_debugger = ModelDebugger(enable_logging=True)



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
    
    # Patch settings
    patch_size: int = 4
    
    # Positional embedding settings
    axes_dim: list[int] = field(default_factory=lambda: [16, 16, 16])  # Default to [temporal_dim, height_dim, width_dim]
    theta: int = 10_000
    
    # Training settings
    use_noise: bool = False
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
            ruleset={pxc.STATUS.INIT: ("h, u <- u:to_init",)},
            tforms={
                "to_init": lambda n, k, v, rkg: jax.random.normal(
                    px.RKG(), (config.num_patches, config.patch_dim)
                ) * 0.01 if config.use_noise else jnp.ones((config.num_patches, config.patch_dim))
            }
        )]
        
        # Create Vodes for each transformer block output
        for _ in range(config.num_blocks):
            self.vodes.append(pxc.Vode(
                ruleset={STATUS_FORWARD: ("h -> u",)}
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
        self.patch_projection = pxnn.Linear(in_features=48, out_features=config.hidden_size)
        
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

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x) # Apply transformer block
            # x = jnp.tanh(x)
            x = self.vodes[i+1](x) # Apply Vode
    
        # Project back to patch_dim
        x = jax.vmap(self.output_projection)(x)
        
        # Apply tanh activation to constrain output values to [-1, 1] range
        # x = jnp.tanh(x)
        
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


@contextmanager
def temp_set_energy_fn(vode, new_energy_fn):
    """Temporarily set a Vode's energy_fn to a px.static-wrapped function."""
    original_energy_fn = vode.energy_fn  # Save the original
    vode.energy_fn = px.static(new_energy_fn)  # Set the new one
    try:
        yield
    finally:
        vode.energy_fn = original_energy_fn  # Restore the original


def masked_se_energy(vode, rkg, mask):
    """
    Compute the masked sensory energy, considering only known pixels.
    
    Args:
        vode: The Vode object containing h and u.
        rkg: Unused (required for energy function signature).
        mask: The mask array, where 1 indicates known pixels and 0 indicates masked pixels.
              Shape should be compatible with h and u (e.g., (1, H, W) will be broadcast).
    
    Returns:
        The masked energy (scalar).
    """
    h = vode.get("h")  # Shape (batch_size, channels, H, W)
    u = vode.get("u")  # Shape (batch_size, channels, H, W)
    
    # Ensure mask has proper broadcasting dimensions for batch processing
    if mask.ndim == 3 and h.ndim == 4:  # If mask lacks batch dimension
        mask = mask[None, ...]  # Add batch dimension: (1, 1, H, W)
        
    error = (h - u) ** 2  # Shape (batch_size, channels, H, W)
    masked_error = error * mask  # Zero out error for masked pixels
    return jnp.sum(masked_error) / jnp.sum(mask)  # Normalize by number of known pixels


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None, step=None):
    model.train()

    h_energy, w_energy, h_grad, w_grad = None, None, None, None

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

    # Top down sweep and setting target value
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference and learning steps
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
        h_energy, w_energy, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step)
        step += 1

    return h_energy, w_energy, h_grad, w_grad


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    # Inference steps
    for t in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)
            # print("h_energy:", h_energy, "at step t:", t)

        optim_h.step(model, h_grad["model"])
    
    optim_h.clear()

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = forward(None, model=model)

    loss = jnp.square(x_hat.flatten() - x.flatten()).mean()

    return loss, x_hat


def eval(dl, T, *, model: TransformerDecoder, optim_h: pxu.Optim):
    losses = []

    for x, y in dl:
        e, y_hat = eval_on_batch(T, x.numpy(), model=model, optim_h=optim_h)
        losses.append(e)

    return jnp.mean(jnp.array(losses))


def eval_on_batch_partial(use_corruption: bool, corrupt_ratio: float, T: int, x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    # model.eval()

    # Extract image shape information statically
    image_shape = model.config.image_shape
    channels, H, W = image_shape
    
    # Create mask for corrupted regions
    corrupt_height = int(corrupt_ratio * H)
    mask = jnp.ones((H, W), dtype=jnp.float32)
    
    if use_corruption:
        # Set bottom part to 0 (corrupted region)
        mask = mask.at[corrupt_height:, :].set(0)
    
    # Broadcast mask to match image dimensions (without batch dim yet)
    mask_broadcasted = mask[None, :, :]  # Shape (1, H, W)
    
    # Apply mask to input image (handle batch dimension if present)
    batch_mode = len(x.shape) == 4  # Check if input is batched
    
    if use_corruption:
        if batch_mode:
            # Batched input - apply to each image
            x_corrupted = x.copy()
            for c in range(channels):
                # For each batch item, corrupt the bottom region
                x_corrupted = x_corrupted.at[:, c, corrupt_height:, :].set(0.0)
        else:
            # Single image
            x_corrupted = x.copy()
            for c in range(channels):
                x_corrupted = x_corrupted.at[c, corrupt_height:, :].set(0.0)
    else:
        x_corrupted = x
    
    # Set energy function for output Vode to use masked energy
    with temp_set_energy_fn(model.vodes[-1], lambda vode, rkg: masked_se_energy(vode, rkg, mask_broadcasted)):
        inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
            energy
        )
        
        # Init step
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(x_corrupted, model=model)
        
        optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
        
        # Inference steps
        for _ in range(T):
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (h_energy, y_), h_grad = inference_step(model=model)
                
            optim_h.step(model, h_grad["model"])
        
        optim_h.clear()
        
        with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
            x_hat = forward(None, model=model)
    
    # If using corruption, compute loss only on corrupted region
    if use_corruption:
        if batch_mode:
            # For batched input, evaluate each item
            x_flat = x[:, :, corrupt_height:, :].reshape(-1)
            x_hat_flat = x_hat[:, :, corrupt_height:, :].reshape(-1)
        else:
            # For single image
            x_flat = x[:, corrupt_height:, :].reshape(-1)
            x_hat_flat = x_hat[:, corrupt_height:, :].reshape(-1)
        
        loss = jnp.square(jnp.clip(x_hat_flat, 0.0, 1.0) - x_flat).mean()
    else:
        # Otherwise, compute loss on the whole image
        loss = jnp.square(jnp.clip(x_hat.reshape(-1), 0.0, 1.0) - x.reshape(-1)).mean()
        
    return loss, x_hat


def visualize_reconstruction(model, optim_h, dataloader, T_values=[24], use_corruption=False, corrupt_ratio=0.5, target_class=None, num_images=2):
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Extract image shape statically
    image_shape = model.config.image_shape
    num_channels = image_shape[0]
    
    orig_images = []
    recon_images = {T: [] for T in T_values}
    labels_list = []
    dataloader_iter = iter(dataloader)
    
    # Add debug information for reconstruction
    debug_info = {'patched_outputs': {}, 'last_layer_outputs': {}}
    
    # Load and reshape images
    for _ in range(num_images):
        x, label = next(dataloader_iter)
        x = jnp.array(x.numpy())
        
        orig_images.append(jnp.reshape(x[0], image_shape))
        
        # Handle label as scalar or 1-element array
        if hasattr(label, 'item'):
            labels_list.append(label[0].item() if len(label.shape) > 0 else label.item())
        else:
            labels_list.append(None)
            
        for T in T_values:
            # Enhanced eval_on_batch_partial to capture intermediate outputs
            x_hat = eval_on_batch_partial(use_corruption=use_corruption, corrupt_ratio=corrupt_ratio, T=T, x=x, model=model, optim_h=optim_h)
                        
            x_hat_single = jnp.reshape(x_hat[1][0], image_shape)
            recon_images[T].append(x_hat_single)
    
    # Create subplots
    fig, axes = plt.subplots(num_images, 1 + len(T_values), figsize=(4 * (1 + len(T_values)), 2 * num_images))
    
    # If num_images = 1, make axes 2D by adding a row dimension
    if num_images == 1:
        axes = axes[None, :]  # Shape becomes (1, 1 + len(T_values))
    
    # Plot images
    for i in range(num_images):
        # Check number of channels
        if num_channels == 1:  # Grayscale
            axes[i, 0].imshow(jnp.clip(jnp.squeeze(orig_images[i]), 0.0, 1.0), cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(jnp.clip(jnp.transpose(orig_images[i], (1, 2, 0)), 0.0, 1.0))
        axes[i, 0].set_title(f'Original {labels_list[i] if labels_list[i] is not None else ""}')
        axes[i, 0].axis('off')
        
        for j, T in enumerate(T_values):
            if num_channels == 1:  # Grayscale
                axes[i, j+1].imshow(jnp.clip(jnp.squeeze(recon_images[T][i]), 0.0, 1.0), cmap='gray')
            else:  # RGB
                axes[i, j+1].imshow(jnp.clip(jnp.transpose(recon_images[T][i], (1, 2, 0)), 0.0, 1.0))
            axes[i, j+1].set_title(f'T={T}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/reconstruction_{timestamp}.png")
    
    # Debug plot for the last layer outputs and unpatchify process
    debug_dir = "../debug_logs/reconstruction"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create visualizations of the last layer outputs
    for T in debug_info['last_layer_outputs']:
        for idx, last_layer_out in enumerate(debug_info['last_layer_outputs'][T]):
            try:
                plt.figure(figsize=(12, 6))
                
                # Visualize the distribution of values
                plt.subplot(1, 2, 1)
                plt.hist(np.array(last_layer_out).flatten(), bins=50)
                plt.title(f"Last Layer Output Distribution (T={T}, img={idx})")
                
                # Visualize a portion of the last layer output as a heatmap
                plt.subplot(1, 2, 2)
                sample_size = min(20, last_layer_out.shape[1])
                plt.imshow(np.array(last_layer_out[0, :sample_size, :sample_size]), cmap='viridis')
                plt.title(f"Last Layer Output Heatmap (sample)")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(f"{debug_dir}/last_layer_T{T}_img{idx}_{timestamp}.png")
                plt.close()
            except Exception as e:
                print(f"Error creating last layer visualization: {e}")
    
    plt.close()
    return orig_images, recon_images


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