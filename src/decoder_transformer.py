import multiprocessing
# Set the start method to 'spawn' instead of 'fork'
multiprocessing.set_start_method('spawn', force=True) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable, List, Optional, Dict, Any
from functools import partial
from contextlib import contextmanager 
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf
from flax import nnx
from jax.typing import DTypeLike

# Import PCX-compatible transformer components
from pcx_transformer import PCXSingleStreamBlock, PCXEmbedND, PCXMLPEmbedder

STATUS_FORWARD = "forward"
STATUS_REFINE = "refine"

import jax.random as jrandom
key = jrandom.PRNGKey(42)  # Same seed in both versions


@dataclass
class TransformerConfig:
    """Configuration for the TransformerDecoder model."""
    # Input/output dimensions
    latent_dim: int = 512
    image_shape: tuple = (3, 32, 32)  # (channels, height, width)
    
    # Architecture settings
    hidden_size: int = 256
    num_heads: int = 8
    num_blocks: int = 3  # Reduced number of blocks for simplicity
    mlp_ratio: float = 4.0
    
    # Patch settings
    patch_size: int = 4  # Size of patches (4x4)
    
    # Training settings
    use_noise: bool = True
    param_dtype: DTypeLike = jnp.float32
    act_fn: Callable[[jax.Array], jax.Array] = jax.nn.gelu
    
    # Positional embedding settings
    theta: int = 10000
    
    # Computed properties
    num_patches: int = field(init=False)
    patch_dim: int = field(init=False)
    axes_dim: List[int] = field(init=False)
    
    def __post_init__(self):
        # Calculate number of patches
        h_patches = self.image_shape[1] // self.patch_size
        w_patches = self.image_shape[2] // self.patch_size
        self.num_patches = h_patches * w_patches
        
        # Calculate patch dimension
        self.patch_dim = self.patch_size * self.patch_size * self.image_shape[0]
        
        # Ensure latent_dim is compatible with number of patches
        if self.latent_dim != self.num_patches * self.patch_dim:
            print(f"Warning: Adjusting latent_dim from {self.latent_dim} to {self.num_patches * self.patch_dim} to match patch dimensions")
            self.latent_dim = self.num_patches * self.patch_dim
        
        # Set up positional embedding dimensions
        self.axes_dim = [self.hidden_size // self.num_heads // 2, self.hidden_size // self.num_heads // 2]


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = px.static(config)
        self.image_shape = px.static(config.image_shape)
        self.patch_size = px.static(config.patch_size)
        self.num_patches = px.static(config.num_patches)
        self.patch_dim = px.static(config.patch_dim)
        
        # Verify patch dimensions match expectation
        h_patches = config.image_shape[1] // config.patch_size
        w_patches = config.image_shape[2] // config.patch_size
        actual_num_patches = h_patches * w_patches
        actual_patch_dim = config.patch_size * config.patch_size * config.image_shape[0]
        
        if config.num_patches != actual_num_patches or config.patch_dim != actual_patch_dim:
            print(f"WARNING: Config patch dimensions don't match calculated dimensions:")
            print(f"Config: num_patches={config.num_patches}, patch_dim={config.patch_dim}")
            print(f"Calculated: num_patches={actual_num_patches}, patch_dim={actual_patch_dim}")
            # Don't override config values as they may be needed for other parts of the code
        
        # Initialize nnx random key
        self.rngs = nnx.Rngs(0)

        # Define Vodes for predictive coding
        self.vodes = [
            # Top-level latent Vode
            pxc.Vode(
                energy_fn=None,
                ruleset={pxc.STATUS.INIT: ("h, u <- u:to_init",)},
                tforms={"to_init": lambda n, k, v, rkg: jrandom.normal(px.RKG(), (config.latent_dim,)) * 0.01 if config.use_noise else jnp.zeros((config.latent_dim,))}
            )
        ]
        
        # Create Vodes for each transformer block output
        for _ in range(config.num_blocks):
            self.vodes.append(
                pxc.Vode(
                    ruleset={STATUS_FORWARD: ("h -> u",)}
                )
            )
        
        # Output Vode (sensory layer)
        self.vodes.append(pxc.Vode())
        
        # Freeze the output Vode's hidden state
        self.vodes[-1].h.frozen = True
        
        # Initialize Transformer components using PCX-compatible wrappers
        
        # Patch embedding projection - use the verified actual_patch_dim
        self.patch_proj = pxnn.Linear(
            in_features=actual_patch_dim,
            out_features=config.hidden_size
        )
        
        # Positional embedding
        self.pe_embedder = PCXEmbedND(
            dim=config.hidden_size // config.num_heads,
            theta=config.theta,
            axes_dim=config.axes_dim
        )
        
        # Create transformer blocks (using SingleStreamBlock instead of DoubleStreamBlock)
        self.transformer_blocks = []
        for i in range(config.num_blocks):
            self.transformer_blocks.append(
                PCXSingleStreamBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    rngs=self.rngs,
                    param_dtype=config.param_dtype
                )
            )
        
        # Output projection from hidden dimension back to patch
        self.out_proj = pxnn.Linear(
            in_features=config.hidden_size,
            out_features=config.patch_dim
        )

    def patchify(self, x):
        """Convert image to sequence of patches.
        
        Args:
            x: Image tensor of shape (channels, height, width) or (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (num_patches, patch_dim) or (batch_size, num_patches, patch_dim)
        """
        # Get patch size and calculate dimensions directly
        p = self.patch_size.get()
        h_patches = self.image_shape.get()[1] // p
        w_patches = self.image_shape.get()[2] // p
        c = self.image_shape.get()[0]
        
        # Check if input is batched
        batch_mode = len(x.shape) == 4
        
        if batch_mode:
            # Handle batched input
            batch_size = x.shape[0]
            # Reshape to extract patches for each batch item
            x = x.reshape((batch_size, c, h_patches, p, w_patches, p))
            x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))  # (batch_size, h_patches, w_patches, c, p, p)
            x = x.reshape((batch_size, h_patches * w_patches, c * p * p))  # (batch_size, num_patches, patch_dim)
        else:
            # Handle single image
            # Reshape to extract patches
            x = x.reshape((c, h_patches, p, w_patches, p))
            x = jnp.transpose(x, (1, 3, 0, 2, 4))  # (h_patches, w_patches, c, p, p)
            x = x.reshape((h_patches * w_patches, c * p * p))  # (num_patches, patch_dim)
        
        return x

    def unpatchify(self, x):
        """Convert sequence of patches back to image.
        
        Args:
            x: Tensor of shape (num_patches, patch_dim) or (batch_size, num_patches, patch_dim)
            
        Returns:
            Image tensor of shape (channels, height, width) or (batch_size, channels, height, width)
        """
        # Get patch size and calculate dimensions directly
        p = self.patch_size.get()
        h_patches = self.image_shape.get()[1] // p
        w_patches = self.image_shape.get()[2] // p
        c = self.image_shape.get()[0]
        
        # Check if input is batched
        batch_mode = len(x.shape) == 3
        
        if batch_mode:
            # Handle batched input
            batch_size = x.shape[0]
            # Reshape back to image for each batch item
            x = x.reshape((batch_size, h_patches, w_patches, c, p, p))
            x = jnp.transpose(x, (0, 3, 1, 4, 2, 5))  # (batch_size, c, h_patches, p, w_patches, p)
            x = x.reshape((batch_size, c, h_patches * p, w_patches * p))  # (batch_size, c, height, width)
        else:
            # Handle single sequence
            # Reshape back to image
            x = x.reshape((h_patches, w_patches, c, p, p))
            x = jnp.transpose(x, (2, 0, 3, 1, 4))  # (c, h_patches, p, w_patches, p)
            x = x.reshape((c, h_patches * p, w_patches * p))  # (c, height, width)
        
        return x

    def generate_grid_coords(self, batch_size):
        """Generate grid coordinates for patches in batch mode.
        
        Args:
            batch_size: The batch size for which to generate coordinates
            
        Returns:
            Grid coordinates with shape (batch_size, num_patches, 2)
        """
        # Calculate patch dimensions directly
        h_patches = self.image_shape.get()[1] // self.patch_size.get()
        w_patches = self.image_shape.get()[2] // self.patch_size.get()
        
        # Create coordinate grid for one item
        coords = jnp.stack(
            jnp.meshgrid(
                jnp.arange(h_patches),
                jnp.arange(w_patches),
                indexing='ij'
            ),
            axis=-1
        ).reshape(-1, 2)  # Shape: (num_patches, 2)
        
        # Repeat the coords for each item in the batch
        coords = jnp.tile(coords[None, ...], (batch_size, 1, 1))
        
        return coords

    def __call__(self, y: jax.Array | None = None):
        # Get the top-level latent from the first Vode
        x = self.vodes[0](jnp.empty(()))  # Shape: (latent_dim,) - NOT batched yet
        
        # Determine batch size from the input y if provided, otherwise default to 1
        if y is not None and hasattr(y, 'shape') and len(y.shape) > 3:
            # If y is provided and has a batch dimension
            batch_size = y.shape[0]
        else:
            # During inference without a provided y, we assume batch_size=1
            # This will be overridden by vmap during training
            batch_size = 1
        
        # Calculate actual dimensions to avoid any mismatches
        h_patches = self.image_shape.get()[1] // self.patch_size.get()
        w_patches = self.image_shape.get()[2] // self.patch_size.get()
        actual_num_patches = h_patches * w_patches
        actual_patch_dim = self.patch_size.get() * self.patch_size.get() * self.image_shape.get()[0]
        
        # Reshape latent to include batch dimension and then sequence of patches
        x = jnp.tile(x[None, ...], (batch_size, 1))  # Shape: (batch_size, latent_dim)
        
        # Verify that the latent dimension is compatible with our patch dimensions
        expected_dim = actual_num_patches * actual_patch_dim
        if x.shape[1] != expected_dim:
            print(f"WARNING: Latent dimension {x.shape[1]} doesn't match expected dimension {expected_dim}")
            print(f"actual_num_patches: {actual_num_patches}, actual_patch_dim: {actual_patch_dim}")
        
        # Reshape using the actual dimensions we just calculated
        x = x.reshape((batch_size, actual_num_patches, actual_patch_dim))
        
        # Generate positional embeddings with batch dimension (using actual patch count)
        coords = self.generate_grid_coords(batch_size)  # Shape: (batch_size, num_patches, 2)
        
        # Project patches to hidden dimension for all items in batch
        x = jax.vmap(self.patch_proj)(x)  # Shape: (batch_size, num_patches, hidden_size)
        
        # Process coordinates through pe_embedder one item at a time to ensure correct shapes
        def process_coords(item_coords):
            # Process a single item's coordinates
            pe = self.pe_embedder(item_coords)  # Shape: (num_patches, 1, pe_dim)
            return pe.squeeze(1)  # Shape: (num_patches, pe_dim)
            
        # Apply the processing to each batch item's coordinates
        pe = jax.vmap(process_coords)(coords)  # Shape: (batch_size, num_patches, pe_dim)
        
        # Create conditioning vector (all zeros)
        vec = jnp.zeros((self.config.hidden_size,))
        
        # Process each sequence through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Create a function that applies the block to a single sequence
            def apply_block(seq, pos_enc):
                return block(seq, vec, pos_enc)
            
            # Apply the transformer block to each sequence with its positional encoding
            x = jax.vmap(apply_block)(x, pe)
            x = self.vodes[i+1](x)  # Apply Vode
            
        # Project each sequence back to patch representation
        x = jax.vmap(self.out_proj)(x)  # Shape: (batch_size, num_patches, patch_dim)
        
        # Reshape each item back to image format
        x = jax.vmap(self.unpatchify)(x)  # Shape: (batch_size, channels, height, width)
        
        # Apply final Vode
        x = self.vodes[-1](x)  # Shape: (batch_size, channels, height, width)
        
        if y is not None:
            # If target image is provided, set the sensory Vode's hidden state
            self.vodes[-1].set("h", y)
            
        return self.vodes[-1].get("u")  # Return the predicted image


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: TransformerDecoder):
    """Forward pass of the model. This function is vmapped for batch processing.
    
    The model's __call__ method now handles batched inputs automatically.
    """
    return model(y=x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: TransformerDecoder):
    """Energy computation for the model. This function is vmapped for batch processing."""
    # Using None as input since we're using the latent from the model's Vode
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
def train_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    h_energy, w_energy, h_grad, w_grad = None, None, None, None

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

    # Top down sweep and setting target value
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference and learning steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (h_energy, y_), h_grad = inference_step(model=model)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (w_energy, y_), w_grad = learning_step(model=model)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])
    
    optim_h.clear()

    return h_energy, w_energy, h_grad, w_grad


def train(dl, T, *, model: TransformerDecoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        h_energy, w_energy, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h)


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: TransformerDecoder, optim_h: pxu.Optim):
    model.eval()

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
    model.eval()

    # Create mask for corrupted regions
    channels, H, W = model.image_shape.get()
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

    orig_images = []
    recon_images = {T: [] for T in T_values}
    labels_list = []
    dataloader_iter = iter(dataloader)
    
    # Load and reshape images
    for _ in range(num_images):
        x, label = next(dataloader_iter)
        x = jnp.array(x.numpy())
        
        orig_images.append(jnp.reshape(x[0], model.image_shape.get()))
        
        # Handle label as scalar or 1-element array
        if hasattr(label, 'item'):
            labels_list.append(label[0].item() if len(label.shape) > 0 else label.item())
        else:
            labels_list.append(None)
            
        for T in T_values:
            x_hat = eval_on_batch_partial(use_corruption=use_corruption, corrupt_ratio=corrupt_ratio, T=T, x=x, model=model, optim_h=optim_h)
            x_hat_single = jnp.reshape(x_hat[1][0], model.image_shape.get())
            recon_images[T].append(x_hat_single)
    
    # Create subplots
    fig, axes = plt.subplots(num_images, 1 + len(T_values), figsize=(4 * (1 + len(T_values)), 2 * num_images))
    
    # If num_images = 1, make axes 2D by adding a row dimension
    if num_images == 1:
        axes = axes[None, :]  # Shape becomes (1, 1 + len(T_values))
    
    # Plot images
    for i in range(num_images):
        # Check number of channels with .get()
        if model.image_shape.get()[0] == 1:  # Grayscale
            axes[i, 0].imshow(jnp.clip(jnp.squeeze(orig_images[i]), 0.0, 1.0), cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(jnp.clip(jnp.transpose(orig_images[i], (1, 2, 0)), 0.0, 1.0))
        axes[i, 0].set_title(f'Original {labels_list[i] if labels_list[i] is not None else ""}')
        axes[i, 0].axis('off')
        
        for j, T in enumerate(T_values):
            if model.image_shape.get()[0] == 1:  # Grayscale
                axes[i, j+1].imshow(jnp.clip(jnp.squeeze(recon_images[T][i]), 0.0, 1.0), cmap='gray')
            else:  # RGB
                axes[i, j+1].imshow(jnp.clip(jnp.transpose(recon_images[T][i], (1, 2, 0)), 0.0, 1.0))
            axes[i, j+1].set_title(f'T={T}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/reconstruction_{timestamp}.png")
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