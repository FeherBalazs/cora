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
from flax import nnx
from jax.typing import DTypeLike
from einops import rearrange
from pcx.core._parameter import get as param_get  # Import get function for parameter access

from pcx_transformer import PCXSingleStreamBlockStandard, PCXEmbedND, PCXMLPEmbedder, PCXLastLayer, PCXLastLayerStandard

STATUS_FORWARD = "forward"
STATUS_REFINE = "refine"

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
    
    # Patch settings
    patch_size: int = 4
    
    # Positional embedding settings
    axes_dim: list[int] = field(default_factory=lambda: [16, 16, 16])  # Default to [temporal_dim, height_dim, width_dim]
    theta: int = 10_000
    
    # Training settings
    use_noise: bool = True
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


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = px.static(config)
        
        # Initialize random key
        self.rngs = nnx.Rngs(0)
        key1, key2 = jax.random.split(self.rngs.default, num=2)

        print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
        print(f"Using {'video' if self.config.is_video else 'image'} mode with shape {self.config.image_shape}")

        # Define Vodes for predictive coding
        # Top-level latent Vode
        self.vodes = [pxc.Vode(
            energy_fn=None,
            ruleset={pxc.STATUS.INIT: ("h, u <- u:to_init",)},
            tforms={
                "to_init": lambda n, k, v, rkg: jax.random.normal(
                    px.RKG(), (config.num_patches, config.patch_dim)
                ) * 0.01 if config.use_noise else jnp.zeros((config.num_patches, config.patch_dim))
            }
        )]
        
        # Create Vodes for each transformer block output
        for _ in range(config.num_blocks):
            self.vodes.append(pxc.Vode(
                energy_fn=pxc.se_energy,
                ruleset={STATUS_FORWARD: ("h -> u",)}
            ))

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(config.num_blocks):
            self.transformer_blocks.append(
                pxnn.TransformerBlock(
                    input_shape=config.hidden_size,
                    hidden_dim=config.hidden_size * config.mlp_ratio,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rate,
                    rkg=px.RKG
                )
            )
        
        # Output Vode (sensory layer) - shape depends on whether we're handling video or images
        self.vodes.append(pxc.Vode(energy_fn=pxc.se_energy))
        self.vodes[-1].h.frozen = True  # Freeze the output Vode's hidden state

        # TODO: This is fixed random embedding, change it later to something better
        self.positional_embedding = jax.random.normal(key1, (config.num_patches, config.hidden_size))
    

    def unpatchify(self, x, patch_size, image_size, channel_size):
        """
        Reconstructs a CIFAR-10 image from a sequence of patch embeddings.

        Args:
            x (array): Transformer output, shape (num_patches, patch_dim)
            patch_size (int): Size of each patch (e.g., 4 for 4x4 patches)
            image_size (int): Size of the image (e.g., 32 for 32x32 images)
            channel_size (int): Number of channels (e.g., 3 for RGB)

        Returns:
            image (array): Reconstructed image, shape (image_size, image_size, channel_size)
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
        # Shape: (64, 4, 4, 3) -> (32, 32, 3)
        image = einops.rearrange(
            x,
            '(h_num w_num) p_h p_w c -> (h_num p_h) (w_num p_w) c',
            h_num=num_patches_per_side,
            w_num=num_patches_per_side
        )

        return image
    
    
    def __call__(self, y: jax.Array | None = None):        
        # Get the initial sequence of patch embeddings from Vode 0
        x = self.vodes[0](jnp.empty(()))
        
        # Add positional embeddings
        x += self.positional_embedding

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x) # Apply transformer block
            x = self.vodes[i+1](x) # Apply Vode
    
        # Apply tanh activation to constrain output values to [-1, 1] range
        # This matches the input normalization range and helps stabilize training
        x = jnp.tanh(x)
        
        # Unpatchify back to image
        x = self.unpatchify(x, patch_size=self.config.patch_size, image_size=self.config.image_shape[1], channel_size=self.config.image_shape[0])
        
        # Apply sensory Vode
        x = self.vodes[-1](x)
        
        # Set target if provided
        if y is not None:
            self.vodes[-1].set("h", y)
        
        return self.vodes[-1].get("u")  # Return prediction


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: TransformerDecoder):
    """Forward pass of the model."""
    return model(y=x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: TransformerDecoder):
    """Energy computation for the model."""
    y_ = model(y=None)
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
        # Capture the forward outputs for debugging
        z = forward(x, model=model)
    
    # Do not log activity stats in JIT-compiled function
    # We'll log stats in the debug_one_batch function outside of JIT

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference and learning steps
    for t in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (w_energy, y_), w_grad = learning_step(model=model)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])
        
        # Energy and gradient logging will happen in a non-JIT context after this function returns

    # After training, forward once more to get final activations
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        z = forward(None, model=model)
    
    # No debugging visualization here - will do that in a non-JIT context

    optim_h.clear()

    return h_energy, w_energy, h_grad, w_grad


def train(dl, T, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim, epoch=None):
    step = 0
    for x, y in dl:
        h_energy, w_energy, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step)
        step += 1
    
    # Save all logs after each epoch
    model_debugger.save_all_logs()


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    # model.eval()

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)

        optim_h.step(model, h_grad["model"])
    
    optim_h.clear()

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = forward(None, model=model)

    loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - x.flatten()).mean()

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
            
            # For debugging, capture last layer outputs
            try:
                with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
                    z = forward(None, model=model)
                    if model.final_layer:
                        # The TransformerDecoder doesn't have a 'latent' attribute directly
                        # Instead, we can get the conditioning parameter
                        cond_param = param_get(model.cond_param)
                        last_layer_out = model.final_layer.module(z, cond_param)
                        # Store for debugging
                        if T not in debug_info['last_layer_outputs']:
                            debug_info['last_layer_outputs'][T] = []
                        debug_info['last_layer_outputs'][T].append(last_layer_out)
            except Exception as e:
                print(f"Error capturing last layer outputs: {e}")
            
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
    # This configuration setup is now compatible with multiple datasets
    config = create_config_by_dataset(
        dataset_name="cifar10",
        latent_dim=512,
        num_blocks=6
    )
    
    # Create model with config
    model = TransformerDecoder(config)