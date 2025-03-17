import multiprocessing
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
from einops import rearrange

# Import PCX-compatible transformer components
from pcx_transformer import PCXSingleStreamBlock, PCXEmbedND, PCXMLPEmbedder, PCXLastLayer

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
    num_blocks: int = 3
    mlp_ratio: float = 4.0
    
    # Patch settings
    patch_size: int = 4  # Size of patches (4x4)
    
    # Positional embedding settings
    axes_dim: list[int] = field(default_factory=lambda: [16, 16])
    theta: int = 10_000
    
    # Training settings
    use_noise: bool = True
    param_dtype: DTypeLike = jnp.float32
    
    def __post_init__(self):
        # Calculate patch dimensions
        self.num_patches = (self.image_shape[1] // self.patch_size) * (self.image_shape[2] // self.patch_size)
        self.patch_dim = self.patch_size * self.patch_size * self.image_shape[0]


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = px.static(config)
        
        # Initialize random key
        self.rngs = nnx.Rngs(0)
        
        print(f"Model initialized with {self.config.num_patches} patches, each with dimension {self.config.patch_dim}")
        
        # Define Vodes for predictive coding
        # Top-level latent Vode
        self.vodes = [pxc.Vode(
            energy_fn=None,
            ruleset={pxc.STATUS.INIT: ("h, u <- u:to_init",)},
            tforms={"to_init": lambda n, k, v, rkg: jax.random.normal(px.RKG(), (config.latent_dim,)) * 0.01 if config.use_noise else jnp.zeros((config.latent_dim,))}
        )]
        
        # Create Vodes for each transformer block output
        for _ in range(config.num_blocks):
            self.vodes.append(pxc.Vode(
                ruleset={STATUS_FORWARD: ("h -> u",)}
            ))
        
        # Output Vode (sensory layer)
        self.vodes.append(pxc.Vode())
        self.vodes[-1].h.frozen = True  # Freeze the output Vode's hidden state
        
        # === jflux-inspired architecture components ===
        
        # Image input projection - using PCX Linear layer properly
        self.img_in = pxnn.Linear(
            in_features=self.config.patch_dim,
            out_features=config.hidden_size,
            bias=True  # Using bias=True for Equinox API
        )
        
        # Latent vector processing via MLPEmbedder
        self.vector_in = PCXMLPEmbedder(
            in_dim=config.latent_dim,
            hidden_dim=config.hidden_size
        )
        
        # Positional embedding generator
        self.pe_embedder = PCXEmbedND(
            dim=config.hidden_size // config.num_heads,
            theta=config.theta,
            axes_dim=config.axes_dim
        )
        
        # Transformer blocks (already contain modulation internally)
        self.transformer_blocks = []
        for i in range(config.num_blocks):
            # Create transformer block
            self.transformer_blocks.append(
                PCXSingleStreamBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    rngs=self.rngs,
                    param_dtype=config.param_dtype
                )
            )
        
        # Use PCXLastLayer for final layer processing
        self.final_layer = PCXLastLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=config.image_shape[0],  # Number of channels
            rngs=self.rngs,
            param_dtype=config.param_dtype
        )

    def _patchify(self, x, batch_size=None):
        """Simple patchify function. Converts image to sequence of patches."""
        c, h, w = self.config.image_shape
        p = self.config.patch_size
        
        if batch_size is None:
            # Handle single image
            x = x.reshape((c, h // p, p, w // p, p))
            x = jnp.transpose(x, (1, 3, 0, 2, 4))  # (h_patches, w_patches, c, p, p)
            x = x.reshape((-1, p * p * c))  # (num_patches, patch_dim)
        else:
            # Handle batched input
            x = x.reshape((batch_size, c, h // p, p, w // p, p))
            x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))  # (batch, h_patches, w_patches, c, p, p)
            x = x.reshape((batch_size, -1, p * p * c))  # (batch, num_patches, patch_dim)
        
        return x
    
    def _unpatchify(self, x, batch_size=None):
        """Simple unpatchify function. Converts patches back to images."""
        c, h, w = self.config.image_shape
        p = self.config.patch_size
        h_patches, w_patches = h // p, w // p
        
        if batch_size is None:
            # Handle single sequence
            x = x.reshape((h_patches, w_patches, c, p, p))
            x = jnp.transpose(x, (2, 0, 3, 1, 4))  # (c, h_patches, p, w_patches, p)
            x = x.reshape((c, h, w))  # (c, h, w)
        else:
            # Handle batched input
            x = x.reshape((batch_size, h_patches, w_patches, c, p, p))
            x = jnp.transpose(x, (0, 3, 1, 4, 2, 5))  # (batch, c, h_patches, p, w_patches, p)
            x = x.reshape((batch_size, c, h, w))  # (batch, c, h, w)
        
        return x
    
    def _create_patch_ids(self, batch_size):
        """Creates patch position IDs for positional embeddings."""
        # Calculate grid dimensions
        h_patches = self.config.image_shape[1] // self.config.patch_size
        w_patches = self.config.image_shape[2] // self.config.patch_size
        
        # Create 2D grid of patch positions
        patch_ids = jnp.zeros((h_patches, w_patches, 2))
        patch_ids = patch_ids.at[..., 0].set(jnp.arange(h_patches)[:, None])
        patch_ids = patch_ids.at[..., 1].set(jnp.arange(w_patches)[None, :])
        
        # Reshape to sequence and add batch dimension
        patch_ids = patch_ids.reshape(-1, 2)
        patch_ids = jnp.tile(patch_ids[None], (batch_size, 1, 1))
        
        return patch_ids

    def __call__(self, y: jax.Array | None = None):
        # Get batch size
        batch_size = 1 if y is None or len(y.shape) < 4 else y.shape[0]
        
        # 1. Get the top-level latent from the first Vode
        latent = self.vodes[0](jnp.empty(()))  # Shape: (latent_dim,)
        
        # 2. Process latent through MLPEmbedder to get conditioning vector
        vec = self.vector_in(latent)  # (hidden_size,)
        
        # Add batch dimension to the conditioning vector for transformer blocks
        vec = vec[None, :]  # Shape: (batch_size=1, hidden_size)
        
        # If we need to repeat for multiple items in the batch
        if batch_size > 1:
            # Replicate the same vector for each item in the batch
            vec = jnp.repeat(vec, batch_size, axis=0)  # Shape: (batch_size, hidden_size)
        
        # 3. Create empty image patches (zeros)
        patch_seq = jnp.zeros((batch_size, self.config.num_patches, self.config.patch_dim))
        
        # 4. Project patches to hidden dimension - ONE SIMPLE STEP
        # Apply linear layer to each patch separately while maintaining batch dimension
        # This double vmap: 
        # - First vmap vectorizes over batch dimension
        # - Second vmap vectorizes over patches within each batch
        patch_dim = self.config.patch_dim
        
        # Create a function that applies the linear layer to a single patch
        def process_single_patch(patch):
            return self.img_in(patch)
        
        # Vectorize across batches and patches
        batched_process = jax.vmap(jax.vmap(process_single_patch))
        
        # Apply to all patches in all batches and ensure shape is (batch_size, num_patches, hidden_size)
        x = batched_process(patch_seq)
        
        # 5. Create position IDs and compute positional embeddings
        patch_ids = self._create_patch_ids(batch_size)
        pe = self.pe_embedder(patch_ids)
        
        # 6. Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Apply transformer block (modulation happens inside)
            x = block(x, vec, pe)
            
            # Apply Vode
            x = self.vodes[i+1](x)
        
        # 7. Apply final layer processing using PCXLastLayer (handles modulation internally)
        x = self.final_layer(x, vec)
        
        # 8. Unpatchify back to image
        x = self._unpatchify(x, batch_size)  # (batch, c, h, w)
        
        # 9. Apply final Vode
        x = self.vodes[-1](x)  # (batch, c, h, w)
        
        # 10. Set target if provided
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