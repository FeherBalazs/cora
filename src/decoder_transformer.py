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
from pcx_transformer import PCXDoubleStreamBlock, PCXEmbedND, PCXMLPEmbedder

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
    hidden_size: int = 512
    num_heads: int = 8
    num_blocks: int = 6
    mlp_ratio: float = 4.0
    
    # Spatial dimensions for transformer
    h_init: int = 8
    w_init: int = 8
    
    # Training settings
    use_noise: bool = True
    param_dtype: DTypeLike = jnp.float32
    act_fn: Callable[[jax.Array], jax.Array] = jax.nn.gelu
    
    # Positional embedding settings
    theta: int = 10000
    
    # Computed properties
    seq_len: int = field(init=False)
    channels_init: int = field(init=False)
    axes_dim: List[int] = field(init=False)
    
    def __post_init__(self):
        self.seq_len = self.h_init * self.w_init
        self.channels_init = self.latent_dim // self.seq_len
        
        # Ensure latent_dim is compatible with spatial dimensions
        if self.channels_init * self.seq_len != self.latent_dim:
            raise ValueError(f"latent_dim {self.latent_dim} must be divisible by h_init * w_init = {self.seq_len}")
        
        # Set up positional embedding dimensions
        self.axes_dim = [self.hidden_size // self.num_heads // 2, self.hidden_size // self.num_heads // 2]


class TransformerDecoder(pxc.EnergyModule):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.image_shape = px.static(config.image_shape)
        self.output_dim = config.image_shape[0] * config.image_shape[1] * config.image_shape[2]
        
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
        
        # Input projection from latent to hidden size
        self.latent_proj = pxnn.Linear(
            in_features=config.channels_init,
            out_features=config.hidden_size
        )
        
        # Positional embedding
        self.pe_embedder = PCXEmbedND(
            dim=config.hidden_size // config.num_heads,
            theta=config.theta,
            axes_dim=config.axes_dim
        )
        
        # Create transformer blocks
        self.transformer_blocks = []
        for i in range(config.num_blocks):
            self.transformer_blocks.append(
                PCXDoubleStreamBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    rngs=self.rngs,
                    param_dtype=config.param_dtype
                )
            )
        
        # Output projection to image space
        self.out_proj = pxnn.Linear(
            in_features=config.hidden_size,
            out_features=config.image_shape[0]  # Output channels
        )

    def __call__(self, y: jax.Array | None = None):
        # Get the top-level latent from the first Vode
        x = self.vodes[0](jnp.empty(()))  # Shape: (latent_dim,)
        
        # Reshape latent to sequence form
        x = x.reshape((self.config.seq_len, self.config.channels_init))  # Shape: (seq_len, channels_init)
        
        # Project to hidden dimension
        x = self.latent_proj(x)  # Shape: (seq_len, hidden_size)
        
        # Generate positional embeddings
        # Create coordinate grid for the sequence
        coords = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.config.h_init),
                jnp.arange(self.config.w_init),
                indexing='ij'
            ),
            axis=-1
        ).reshape(-1, 2)  # Shape: (seq_len, 2)
        
        pe = self.pe_embedder(coords)  # Shape: (seq_len, 1, pe_dim)
        pe = pe.squeeze(1)  # Shape: (seq_len, pe_dim)
        
        # Create dummy text stream (all zeros)
        txt = jnp.zeros((self.config.seq_len, self.config.hidden_size))
        
        # Create dummy vector conditioning (all zeros)
        vec = jnp.zeros((self.config.hidden_size,))
        
        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x, txt = block(x, txt, vec, pe)
            x = self.vodes[i+1](x)  # Apply Vode after each transformer block
        
        # Project to output channels and reshape to image dimensions
        x = self.out_proj(x)  # Shape: (seq_len, channels)
        
        # Reshape to image format (channels, height, width)
        x = x.reshape((self.config.h_init, self.config.w_init, self.image_shape[0]))  # Shape: (h, w, channels)
        x = jnp.transpose(x, (2, 0, 1))  # Shape: (channels, h, w)
        
        # Use spatial upsampling to reach target image size
        if self.config.h_init != self.image_shape[1] or self.config.w_init != self.image_shape[2]:
            # Use simple resize to desired dimensions
            x = jax.image.resize(
                x,
                shape=(self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                method='bilinear'
            )
        
        # Apply final Vode
        x = self.vodes[-1](x)  # Shape: (channels, height, width)
        
        if y is not None:
            # If target image is provided, set the sensory Vode's hidden state
            self.vodes[-1].set("h", y)  # y should be (channels, height, width)
            
        return self.vodes[-1].get("u")  # Return the predicted image


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
              Shape should be compatible with h and u (e.g., (1, 1, H, W)).
    
    Returns:
        The masked energy (scalar).
    """
    h = vode.get("h")  # Shape (batch_size, channels, H, W)
    u = vode.get("u")  # Shape (batch_size, channels, H, W)
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
    channels, H, W = model.image_shape
    corrupt_height = int(corrupt_ratio * H)
    mask = jnp.ones((H, W), dtype=jnp.float32)
    
    if use_corruption:
        # Set bottom part to 0 (corrupted region)
        mask = mask.at[corrupt_height:, :].set(0)
    
    # Broadcast mask to match image dimensions
    mask_broadcasted = mask[None, :, :]  # Shape (1, H, W)
    
    # Apply mask to input image
    if use_corruption:
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
        x_flat = x[:, corrupt_height:, :].flatten()
        x_hat_flat = x_hat[:, corrupt_height:, :].flatten()
        loss = jnp.square(jnp.clip(x_hat_flat, 0.0, 1.0) - x_flat).mean()
    else:
        # Otherwise, compute loss on the whole image
        loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - x.flatten()).mean()
        
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
            h_init=8,
            w_init=8
        )
    elif dataset_name == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            hidden_size=256,
            num_heads=8,
            num_blocks=num_blocks,
            h_init=8,
            w_init=8
        )
    elif dataset_name == "imagenet":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 224, 224),
            hidden_size=384,
            num_heads=8,
            num_blocks=num_blocks,
            h_init=16,
            w_init=16
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