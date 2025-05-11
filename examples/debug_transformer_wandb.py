import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp
import optax
import pcx
import pcx.utils as pxu
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import time
from jax.profiler import trace, StepTraceAnnotation
import json
from datetime import datetime
import wandb
import psutil
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import re
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Callable
import imageio

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import the TransformerDecoder and utility functions
from src.decoder_transformer import (
    TransformerDecoder, 
    TransformerConfig,
    train, # Returns avg_loss, h_grad, w_grad
    # eval, # Replaced by eval_pretext_metrics
    unmask_on_batch,
    unmask_on_batch_enhanced, # Returns loss, recons, mask
    forward,
    eval_pretext_metrics, # New evaluation function
    calculate_psnr # Import if needed elsewhere, though eval_pretext_metrics uses it internally
)

#TODO: simplify config across scripts
@dataclass
class ModelConfig:
    """Configuration for transformer models with all hyperparameters in one place."""
    name: str
    # Dataset settings
    dataset: str = "cifar10"
    data_dir: str = "../datasets/"
    train_subset: int = 50000
    test_subset: int = 1000
    target_class: Optional[int] = None
    reconstruction_every_n_epochs: int = 25 # WARNING: changing this to 1 caused training instability. Less frequent reconstruction is better. Tested only with 10 so far which works ok.
    validation_every_n_epochs: int = 25

    use_corruption: bool = False
    corrupt_ratio: float = 0.25

    use_lower_half_mask: bool = False #If False it uses random masking
    inference_clamp_alpha: float = 0.5

    # Visualization settings
    num_images: int = 2
    
    # Model architecture
    hidden_size: int = 48
    num_heads: int = 6
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    patch_size: int = 4
    axes_dim: List[int] = field(default_factory=lambda: [16, 16])
    theta: int = 100
    act_fn: Callable = jax.nn.swish

    # Status init settings for training and unmasking
    use_status_init_in_training: bool = False
    use_status_init_in_unmasking: bool = False
    
    # Training settings
    use_noise: bool = True
    batch_size: int = 200
    epochs: int = 25
    inference_steps: int = 20
    eval_inference_steps: List[int] = field(default_factory=lambda: [20])
    reconstruction_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 8, 12, 16, 20, 40])

    # # Settings without status.init: epochs=10, hidden_size=64, num_blocks=0, inference_steps=24, mse=0.007
    # peak_lr_weights: float = 0.005
    # peak_lr_hidden: float = 0.0075

    # Settings without status.init: epochs=20, hidden_size=64, num_blocks=1, inference_steps=20, update_weights_every_inference_step=False, mse=0.0008
    peak_lr_weights: float = 0.001
    peak_lr_hidden: float = 0.1

    # # Settings without status.init: hidden_size=64, num_blocks=3, inference_steps=24
    # peak_lr_weights: float = 0.0025
    # peak_lr_hidden: float = 0.0025

    # # Settings with status.init - general
    # peak_lr_weights: float = 0.0001
    # peak_lr_hidden: float = 0.005

    update_weights_during_unmasking: bool = False

    hidden_lr_inference: float = peak_lr_hidden * 1
    weight_decay: float = 2e-4
    warmup_epochs: int = 5
    use_lr_schedule: bool = False
    seed: int = 42
    
    # Layer-specific inference LR scaling
    # TODO: It is not working yet
    use_inference_lr_scaling: bool = False # Enable/disable scaling
    inference_lr_scale_lower: float = 10.0  # Multiplier for lower layers (Vodes < boundary)
    inference_lr_scale_upper: float = 1.0  # Multiplier for upper layers (Vodes >= boundary)
    inference_lr_scale_boundary: int = 4   # Index separating lower/upper (e.g., 3 means 0,1,2 are lower)
    
    # Early stopping settings
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    save_reconstruction_images: bool = True # Option to save static image grid
    save_reconstruction_video: bool = True # Option to save video
    video_fps: int = 60 # Frames per second for the reconstruction video
    reinitialize_model_for_each_epoch: bool = False # WARNING: setting this to True will get around 0.24 val loss with 100 images vs 0.15 without.
    
    update_weights_every_inference_step: bool = False # New flag for training mode


# Predefined configurations for easy experimentation
MODEL_CONFIGS = {
    "debug_tiny": ModelConfig(
        name="debug_tiny",
        hidden_size=64,
        num_heads=1,
        num_blocks=1,
    ),
    "debug_small": ModelConfig(
        name="debug_small",
        hidden_size=512,
        num_heads=8,
        num_blocks=6,
    ),
    "baseline": ModelConfig(
        name="baseline",
        hidden_size=128,
        num_heads=8,
        num_blocks=6,
    )
}

# Default configuration to use
DEFAULT_CONFIG = "debug_small"


def create_config(dataset="cifar10", hidden_size=48, num_blocks=1, num_heads=6,
                 mlp_ratio=4.0, patch_size=4, axes_dim=None, theta=10_000, use_noise=True, use_lower_half_mask=False,
                 use_inference_lr_scaling=False, inference_lr_scale_lower=1.0, inference_lr_scale_upper=1.0, inference_lr_scale_boundary=3,
                 inference_clamp_alpha=1.0, update_weights_during_unmasking=False,
                 use_status_init_in_training: bool = True, use_status_init_in_unmasking: bool = True,
                 update_weights_every_inference_step: bool = True):
    """Create a TransformerConfig based on the dataset name and parameters."""
    axes_dim = axes_dim or [16, 16]
    
    if dataset == "cifar10":
        return TransformerConfig(
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            axes_dim=axes_dim,
            theta=theta,
            use_noise=use_noise,
            use_lower_half_mask=use_lower_half_mask,
            use_inference_lr_scaling=use_inference_lr_scaling,
            inference_lr_scale_lower=inference_lr_scale_lower,
            inference_lr_scale_upper=inference_lr_scale_upper,
            inference_lr_scale_boundary=inference_lr_scale_boundary,
            inference_clamp_alpha=inference_clamp_alpha,
            update_weights_during_unmasking=update_weights_during_unmasking,
            use_status_init_in_training=use_status_init_in_training,
            use_status_init_in_unmasking=use_status_init_in_unmasking,
            update_weights_every_inference_step=update_weights_every_inference_step
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Debug a transformer model with W&B logging')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help=f'Predefined configuration to use. Options: {", ".join(MODEL_CONFIGS.keys())}')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for')
    parser.add_argument('--num_blocks', type=int, default=None,
                        help='Number of transformer blocks')
    parser.add_argument('--peak_lr_weights', type=float, default=None,
                        help='Peak learning rate for weights')
    parser.add_argument('--peak_lr_hidden', type=float, default=None,
                        help='Peak learning rate for hidden states')
    parser.add_argument('--save_reconstruction_images', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Save reconstruction images (true/false)')
    parser.add_argument('--save_reconstruction_video', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Save reconstruction video (true/false)')
    # Add any other parameters from ModelConfig you want to control via CLI here
    return parser.parse_args()


def create_learning_rate_schedule(base_lr, warmup_epochs, total_epochs, steps_per_epoch):
    """Create a learning rate schedule with warmup and cosine decay to half of the peak rate."""
    # Set minimum learning rate to half of the peak
    min_lr = base_lr * 0.5
    
    def lr_schedule(step):
        epoch = step / steps_per_epoch
        
        # Linear warmup phase
        warmup_lr = base_lr * (epoch / warmup_epochs)
        
        # Cosine decay phase with minimum value = min_lr
        decay_ratio = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0)  # Ensure it's between 0 and 1
        
        # Modified cosine decay to range from base_lr to min_lr (not zero)
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        decay_lr = min_lr + (base_lr - min_lr) * cosine_factor
        
        # Return warmup_lr during warmup, decay_lr afterward
        return jnp.where(epoch < warmup_epochs, warmup_lr, decay_lr)
    
    return lr_schedule


def get_debug_dataloaders(dataset_name, batch_size, root_path, train_subset_n=None, test_subset_n=None, target_class=None):
    """Get data loaders with simple augmentation for debugging."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_root = '../' + root_path + "/cifar10/"
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=train_transform,
        download=True,
        train=True,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=test_transform,
        download=True,
        train=False,
    )
    # # TODO: Remove this once we have a proper test set
    # test_dataset = torchvision.datasets.CIFAR10(
    #     root=dataset_root,
    #     transform=test_transform,
    #     download=True,
    #     train=True,
    # )
    
    if target_class is not None:
        target_indices = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        train_dataset = Subset(train_dataset, target_indices)
        target_indices = (test_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        test_dataset = Subset(test_dataset, target_indices)
    
    if train_subset_n is not None:
        all_idx = list(range(len(train_dataset)))
        train_dataset = Subset(train_dataset, all_idx[:train_subset_n])
    if test_subset_n is not None:
        all_idx = list(range(len(test_dataset)))
        test_dataset = Subset(test_dataset, all_idx[:test_subset_n])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    class TorchDataloader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
        
        def __iter__(self):
            return iter(self.dataloader)
        
        def __len__(self):
            return len(self.dataloader)
    
    return TorchDataloader(train_dataloader), TorchDataloader(test_dataloader)


def create_reconstruction_images(intermediate_recons, T_values, orig_images, masked_images, labels_list, num_images, image_shape, wandb_run, epoch):
    """
    Creates a grid of images comparing originals, masked inputs, and reconstructions at specific T_values.
    """
    num_channels, H, W = image_shape
    # Increase columns by 1 to accommodate the masked image
    fig, axes = plt.subplots(num_images, 2 + len(T_values), figsize=(4 * (2 + len(T_values)), 2 * num_images))
    
    if num_images == 1:
        axes = axes[None, :] # Ensure axes is 2D even for one image

    for i in range(num_images):
        # Plot original (Column 0)
        orig_np = np.array(orig_images[i])
        if num_channels == 1:
            axes[i, 0].imshow(np.clip(np.squeeze(orig_np), 0.0, 1.0), cmap='gray')
        else:
            axes[i, 0].imshow(np.clip(np.transpose(orig_np, (1, 2, 0)), 0.0, 1.0))
        axes[i, 0].set_title(f'Original {labels_list[i] if labels_list[i] is not None else ""}')
        axes[i, 0].axis('off')
        
        # Plot masked input (Column 1)
        masked_np = np.array(masked_images[i])
        if num_channels == 1:
            axes[i, 1].imshow(np.clip(np.squeeze(masked_np), 0.0, 1.0), cmap='gray')
        else:
            axes[i, 1].imshow(np.clip(np.transpose(masked_np, (1, 2, 0)), 0.0, 1.0))
        axes[i, 1].set_title(f'Masked Input')
        axes[i, 1].axis('off')

        # Plot reconstructions for specific T values (Start from Column 2)
        for j, T in enumerate(T_values):
            col_idx = j + 2 # Shift column index for reconstructions
            if T in intermediate_recons and i < len(intermediate_recons[T]):
                recon_T = intermediate_recons[T][i] # Get the i-th image for this T
                recon_np = np.array(recon_T) # Assuming it's already (C, H, W) or similar after processing
                
                if num_channels == 1:
                    axes[i, col_idx].imshow(np.clip(np.squeeze(recon_np), 0.0, 1.0), cmap='gray')
                else:
                     # Ensure correct shape and transpose if needed
                    recon_np_reshaped = np.reshape(recon_np, image_shape)
                    axes[i, col_idx].imshow(np.clip(np.transpose(recon_np_reshaped, (1, 2, 0)), 0.0, 1.0))
                axes[i, col_idx].set_title(f'T={T}')
            else:
                 # Handle missing reconstruction for this T/image index
                axes[i, col_idx].set_title(f'T={T} (N/A)')
                axes[i, col_idx].imshow(np.zeros((H, W) if num_channels == 1 else (H, W, 3))) # Placeholder
            axes[i, col_idx].axis('off')

    plt.tight_layout()
    
    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../results", exist_ok=True)
    reconstruction_path = f"../results/reconstruction_images{epoch_str}_{timestamp}.png"
    plt.savefig(reconstruction_path)
    plt.close(fig)
    print(f"Saved reconstruction image grid to {reconstruction_path}")

    log_dict = {}
    if wandb_run is not None:
        log_key = f"reconstructions_images{epoch_str}" if epoch is not None else "reconstructions_images"
        try:
            log_dict[log_key] = wandb.Image(reconstruction_path)
        except Exception as e:
            print(f"Error creating wandb.Image: {e}")

    return reconstruction_path, log_dict


def visualize_reconstruction(model, model_config, optim_h, dataloader, config, wandb_run=None, epoch=None, step=None, optim_w=None):
    
    # Extract static info from config
    image_shape = model.config.image_shape
    num_channels = image_shape[0]
    num_images = config.num_images
    use_corruption = config.use_corruption
    corrupt_ratio = config.corrupt_ratio
    T_values = config.reconstruction_steps # Use T_values from config

    orig_images = []
    recon_images_at_T = {T: [] for T in T_values} # For static images
    all_reconstruction_frames = [] # To store frames for video
    labels_list = []
    masked_images_list = [] # List to store the masked input images
    all_inference_grads = [] # Add list to store grad logs per image
    dataloader_iter = iter(dataloader)

    # Get the single batch
    try:
        x, label = next(dataloader_iter)
        x = jnp.array(x)

        # Process each image in the batch separately
        for i in range(num_images):
            single_x = x[i:i+1]
            orig_images.append(jnp.reshape(single_x[0], image_shape))
            
            if hasattr(label, 'item'):
                labels_list.append(label[i].item() if len(label.shape) > 0 else label.item())
            else:
                labels_list.append(None)
                
            # Call unmask_on_batch_enhanced - capture the mask now
            loss, all_recons_list, mask_single, inference_grads = unmask_on_batch_enhanced(
                use_corruption=use_corruption,
                corrupt_ratio=corrupt_ratio,
                target_T_values=T_values, # Determines max_T internally
                x=single_x,               
                model=model, 
                optim_h=optim_h, # Use optim_h_inference if defined/desired
                optim_w=optim_w, # <<< Pass optim_w
                log_inference_grads=True # Set the flag
            )
            all_inference_grads.append(inference_grads) # Store the logs

            # --- Regenerate the masked input for visualization using the returned mask --- 
            x_init_single = single_x # Default if no corruption
            if use_corruption:
                # Always use the first slice of the returned mask (handles both lower-half and random)
                mask_for_viz = mask_single[0:1] # Shape (1, 1, H, W) 
                # Ensure mask_for_viz is broadcastable for the where clause
                mask_viz = jnp.broadcast_to(mask_for_viz, single_x.shape) 
                # Use gray (0.5) for masked areas for visualization consistency
                placeholder_viz = jnp.ones_like(single_x) * 0.5 
                x_init_single = jnp.where(mask_viz == 1, placeholder_viz, single_x)

            masked_images_list.append(jnp.reshape(x_init_single[0], image_shape))
            # --- End regeneration ---

            # Always store all frames, as we might need them for video
            all_reconstruction_frames.append(all_recons_list)

            # Extract frames needed for static images only if required
            if config.save_reconstruction_images:
                for T_idx, recon in enumerate(all_recons_list):
                    T_step = T_idx + 1 # T steps are 1-indexed
                    if T_step in T_values:
                        # Extract the first image from the batch dimension
                        x_hat_for_T = recon[0] 
                        x_hat_single = jnp.reshape(x_hat_for_T, image_shape) 
                        recon_images_at_T[T_step].append(x_hat_single)

    except StopIteration:
        print("Warning: No data available in dataloader")
        return None, {} # Return None for path, empty dict for logs
    
    # --- Create Visualizations based on config ---
    final_path = None
    combined_log_dict = {}

    # --- Process and log inference gradients ---
    if all_inference_grads:
        print("Logging inference gradients...")
        # Log grads for the first image only for simplicity
        first_image_inference_grads = all_inference_grads[0]
        inference_grad_data = []
        if first_image_inference_grads:
            num_vodes = len(model.vodes)
            # Identify Vodes for transformer blocks (indices 2 to num_blocks + 1)
            # Also include Vode 0 (initial patch state) and Vode 1 (projected patches)
            block_indices = list(range(2, model_config.num_blocks + 2))
            relevant_vode_indices = [0, 1] + block_indices + [num_vodes - 1] # Add sensory Vode too

            steps = sorted(first_image_inference_grads.keys())
            for t in steps:
                step_norms = first_image_inference_grads.get(t, {}) # Get norms for step t
                for vode_idx in relevant_vode_indices:
                    norm_key = f"vode_{vode_idx}_grad_norm"
                    norm_value = step_norms.get(norm_key, float('nan'))
                    if not np.isnan(norm_value):
                        inference_grad_data.append([t + 1, norm_value, f"Vode {vode_idx}"]) # Use t+1 for 1-based step

        # Log as a multi-line plot using wandb's built-in functionality
        if inference_grad_data:
            print("Logging inference gradient time series plot...")
            try:
                log_key = f"InferenceGradients/TimeSeries_Epoch{epoch}"
                # Create plot using custom helper function
                custom_plot = create_multi_line_chart(
                    table_data=inference_grad_data,
                    x_col="inference_step",
                    y_col="grad_norm",
                    series_col="vode",
                    title=f"Inference Gradient Norms vs. Step T (Epoch {epoch})"
                )
                # Log the custom plot HTML object directly
                wandb.log({log_key: custom_plot}, step=epoch)
                print(f"Logged inference gradient time series plot for W&B (key: {log_key}).")
            except Exception as e:
                print(f"Error creating W&B line series plot for inference gradients: {e}")
        else:
            print("No valid inference gradient data to log.")

    # Create static image grid if requested
    if config.save_reconstruction_images:
        image_path, image_log_dict = create_reconstruction_images(
            intermediate_recons=recon_images_at_T,
            T_values=T_values,
            orig_images=orig_images,
            masked_images=masked_images_list,
            labels_list=labels_list,
            num_images=num_images,
            image_shape=image_shape,
            wandb_run=wandb_run,
            epoch=epoch
        )
        final_path = image_path # Prioritize image path if both are created
        combined_log_dict.update(image_log_dict)

    # Create video if requested
    if config.save_reconstruction_video:
        if not all_reconstruction_frames:
            print("Warning: No reconstruction frames generated for video.")
        else:
            video_path, video_log_dict = create_reconstruction_video(
                all_reconstruction_frames=all_reconstruction_frames,
                orig_images=orig_images,
                masked_images=masked_images_list, # Pass masked images
                labels_list=labels_list,
                num_images=num_images,
                image_shape=image_shape,
                wandb_run=wandb_run,
                epoch=epoch,
                fps=config.video_fps # Use fps from config
            )
            if final_path is None:
                final_path = video_path # Set video path if images weren't created
            combined_log_dict.update(video_log_dict)

    return final_path, combined_log_dict


def create_reconstruction_video(all_reconstruction_frames, orig_images, masked_images, labels_list, num_images, image_shape, wandb_run, epoch, fps=10):
    """
    Creates a video comparing original images and their reconstructions over time.

    Args:
        all_reconstruction_frames: List (num_images) of lists (T_steps) of reconstruction tensors.
        orig_images: List of original image tensors.
        masked_images: List of masked input image tensors.
        labels_list: List of labels for the original images.
        num_images: Number of images to include in the video.
        image_shape: Shape of a single image (C, H, W).
        wandb_run: Wandb run object.
        epoch: Current epoch number.
        fps: Frames per second for the video.

    Returns:
        Tuple: (video_path, log_dict)
    """
    num_channels, H, W = image_shape
    num_steps = len(all_reconstruction_frames[0]) # Assuming all images have same number of steps

    video_frames = []

    # Prepare original images once
    processed_orig_images = []
    processed_masked_images = []
    for i in range(num_images):
        orig_np = np.array(orig_images[i])
        masked_np = np.array(masked_images[i])
        if num_channels == 1: # Grayscale
            orig_plot = np.clip(np.squeeze(orig_np), 0.0, 1.0)
            orig_plot = (plt.cm.gray(orig_plot)[:, :, :3] * 255).astype(np.uint8) # Convert grayscale to RGB for video
            masked_plot = np.clip(np.squeeze(masked_np), 0.0, 1.0)
            masked_plot = (plt.cm.gray(masked_plot)[:, :, :3] * 255).astype(np.uint8)
        else: # RGB
            orig_plot = np.clip(np.transpose(orig_np, (1, 2, 0)), 0.0, 1.0)
            orig_plot = (orig_plot * 255).astype(np.uint8)
            masked_plot = np.clip(np.transpose(masked_np, (1, 2, 0)), 0.0, 1.0)
            masked_plot = (masked_plot * 255).astype(np.uint8)
        processed_orig_images.append(orig_plot)
        processed_masked_images.append(masked_plot)

    # Generate frames for the video
    for t in range(num_steps):
        fig, axes = plt.subplots(num_images, 3, figsize=(4 * 3, 2 * num_images))
        if num_images == 1:
            axes = axes[None, :] # Make it 2D

        for i in range(num_images):
            # Plot original image (Column 0)
            axes[i, 0].imshow(processed_orig_images[i])
            axes[i, 0].set_title(f'Original {labels_list[i] if labels_list[i] is not None else ""}')
            axes[i, 0].axis('off')

            # Plot masked input (Column 1)
            axes[i, 1].imshow(processed_masked_images[i])
            axes[i, 1].set_title(f'Masked Input')
            axes[i, 1].axis('off')

            # Plot reconstruction at step t (Column 2)
            recon_t = all_reconstruction_frames[i][t]
            recon_np = np.array(recon_t[0]) # Get the first element from batch dim
            recon_np = np.reshape(recon_np, image_shape)

            if num_channels == 1: # Grayscale
                recon_plot = np.clip(np.squeeze(recon_np), 0.0, 1.0)
                recon_plot = (plt.cm.gray(recon_plot)[:, :, :3] * 255).astype(np.uint8)
            else: # RGB
                recon_plot = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)
                recon_plot = (recon_plot * 255).astype(np.uint8)
            
            # Plot in column 2
            axes[i, 2].imshow(recon_plot)
            axes[i, 2].set_title(f'Recon T={t+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw() # Draw the canvas, cache the renderer
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

        # Calculate shape based on figure size and DPI for robustness
        width_inches, height_inches = fig.get_size_inches()
        dpi = fig.dpi
        width_pixels = int(np.round(width_inches * dpi))
        height_pixels = int(np.round(height_inches * dpi))
        expected_shape = (height_pixels, width_pixels, 3)

        # Check if the buffer size matches the calculated shape
        if frame.size != np.prod(expected_shape):
            print(f"Warning: Buffer size ({frame.size}) does not match calculated shape {expected_shape} ({np.prod(expected_shape)}). Trying to infer shape.")
            # Fallback: Infer shape if calculation doesn't match buffer size
            buffer_pixels = frame.size // 3
            # Ensure fig_width_pixels and fig_height_pixels are defined (from figsize)
            # figsize is (width, height) in inches
            fig_width_inches, fig_height_inches = fig.get_size_inches()
            aspect_ratio = fig_width_inches / fig_height_inches if fig_height_inches > 0 else 1
            # Infer height based on aspect ratio: W = H * aspect_ratio
            # W * H = buffer_pixels => (H * aspect_ratio) * H = buffer_pixels
            # H^2 = buffer_pixels / aspect_ratio
            inferred_height = int(np.sqrt(buffer_pixels / aspect_ratio))
            inferred_width = int(inferred_height * aspect_ratio)

            # Check if inferred shape matches buffer size
            if inferred_height * inferred_width * 3 == frame.size:
                 expected_shape = (inferred_height, inferred_width, 3)
                 print(f"Using inferred shape: {expected_shape}")
            else:
                 print(f"Error: Cannot determine correct frame shape. Buffer size: {frame.size}, Calculated shape: {(height_pixels, width_pixels, 3)}, Inferred shape: {(inferred_height, inferred_width, 3)}")
                 # Raise the original error or handle appropriately
                 raise ValueError(f"Cannot reshape array of size {frame.size} into calculated shape {(height_pixels, width_pixels, 3)} or inferred shape {(inferred_height, inferred_width, 3)}")

        frame = frame.reshape(expected_shape)
        video_frames.append(frame)
        plt.close(fig)

    # Save video
    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../results", exist_ok=True)
    video_path = f"../results/reconstruction_video{epoch_str}_{timestamp}.mp4" # Save as MP4
    imageio.mimsave(video_path, video_frames, fps=fps)
    print(f"Saved reconstruction video to {video_path}")

    # Prepare log dictionary for W&B
    log_dict = {}
    if wandb_run is not None:
        log_key = f"reconstructions_video{epoch_str}" if epoch is not None else "reconstructions_video"
        try:
            log_dict[log_key] = wandb.Video(video_path, fps=fps, format="mp4")
        except Exception as e:
            print(f"Error creating wandb.Video: {e}")
            print("Ensure ffmpeg is installed and accessible.")

    return video_path, log_dict


def log_vode_stats(model, h_grad, w_grad, run, epoch):
    """
    Log statistics about each Vode's energy and gradients to wandb.
    
    Args:
        model: The TransformerDecoder model
        h_grad: Hidden state gradients
        w_grad: Weight gradients
        run: Wandb run object
        epoch: Current epoch
        
    Returns:
        A tuple of (processed_energy_data, processed_grad_data) for summary tables
    """
    # Collect stats for wandb
    vode_stats = {}
    
    # Get energy and gradient data
    vode_energies = model.get_submodule_energies()
    vode_grad_norms = extract_vode_gradient_norms(h_grad)
    
    # Process data for individual metrics and summary tables
    processed_energy_data = []
    processed_grad_data = []
    energy_data = []
    grad_data = []
    
    # Process each vode's data
    for i, (energy, grad_norm) in enumerate(zip(vode_energies, vode_grad_norms)):
        # Process energy values
        if energy is not None:
            if hasattr(energy, 'size') and energy.size == 1:
                energy_value = energy.item()
            elif isinstance(energy, (float, int)):
                # Handle cases where it might already be a Python scalar (less likely from JAX)
                energy_value = float(energy)
            elif isinstance(energy, jax.Array) and energy.size > 1:
                # Handle case where energy is a non-scalar JAX array (e.g., per-batch energies)
                # print(f"Info: Energy for vode {i} is an array (shape: {energy.shape}). Logging mean energy.")
                energy_value = jnp.mean(energy).item() # Calculate and log the mean
            else:
                # Handle unexpected non-scalar or non-array case
                print(f"Warning: Energy for vode {i} is not a scalar JAX array or Python number (type: {type(energy)}, value: {energy}). Logging NaN.")
                energy_value = float('nan') # Assign NaN or another placeholder

            # Log individual metrics (now energy_value is always defined)
            # Check for NaN before adding to data lists used for tables/plots
            if not np.isnan(energy_value):
                 vode_stats[f"VodeEnergy/vode_{i}"] = energy_value
                 energy_data.append([i, energy_value])
                 processed_energy_data.append([epoch+1, i, energy_value])
            else:
                 # Optionally log the NaN value to wandb if desired, or skip
                 vode_stats[f"VodeEnergy/vode_{i}"] = energy_value # Log NaN to wandb
        
        # Process gradient values
        if grad_norm is not None:
            vode_stats[f"VodeGradNorm/vode_{i}"] = grad_norm
            grad_data.append([i, grad_norm])
            processed_grad_data.append([epoch+1, i, grad_norm])
    
    # Create visualizations
    if energy_data:
        energy_table = wandb.Table(data=energy_data, columns=["vode_index", "energy"])
        vode_stats[f"VodeEnergy/distribution_epoch_{epoch+1}"] = wandb.plot.bar(
            energy_table, "vode_index", "energy", 
            title=f"Energy by Vode - Epoch {epoch+1}"
        )
    
    if grad_data:
        grad_table = wandb.Table(data=grad_data, columns=["vode_index", "grad_norm"])
        vode_stats[f"VodeGradNorm/distribution_epoch_{epoch+1}"] = wandb.plot.bar(
            grad_table, "vode_index", "grad_norm", 
            title=f"Gradient Norm by Vode - Epoch {epoch+1}"
        )
    
    # Log to wandb
    run.log(vode_stats, step=epoch+1)
    
    return processed_energy_data, processed_grad_data

def extract_vode_gradient_norms(h_grad):
    """Extract actual gradient L2 norms from the vode gradients"""
    vode_grad_norms = []
    
    # # Add debug information about the structure
    # print(f"h_grad type: {type(h_grad)}")
    # if isinstance(h_grad, dict) and 'model' in h_grad:
    #     print(f"model_grads type: {type(h_grad['model'])}")
    #     if hasattr(h_grad['model'], 'vodes'):
    #         print(f"vodes type: {type(h_grad['model'].vodes)}")
    #         print(f"Number of vodes: {len(h_grad['model'].vodes)}")
    #         if len(h_grad['model'].vodes) > 0:
    #             first_vode = h_grad['model'].vodes[0]
    #             print(f"First vode type: {type(first_vode)}")
    #             if hasattr(first_vode, 'h'):
    #                 print(f"First vode.h type: {type(first_vode.h)}")
    #                 if hasattr(first_vode.h, 'value'):
    #                     print(f"First vode.h.value type: {type(first_vode.h.value)}")
    #                     print(f"First vode.h.value shape: {first_vode.h.value.shape}")
    
    try:
        # Access the model gradients
        model_grads = h_grad['model']
        
        # Access the vodes list
        vodes = model_grads.vodes
        
        # For each vode, extract the gradient tensor and calculate its L2 norm
        for i in range(len(vodes)):
            vode = vodes[i]
            h_grad_tensor = vode.h
            # print("h_grad_tensor", h_grad_tensor)
            
            # VodeParams might store their values in different ways
            # Try different common patterns to access the values
            if hasattr(h_grad_tensor, 'value'):
                # Access the value attribute if it exists
                grad_values = h_grad_tensor.value
            elif hasattr(h_grad_tensor, 'get_value'):
                # Or try get_value() method if it exists
                grad_values = h_grad_tensor.get_value()
            else:
                # Otherwise assume the object itself is the tensor
                grad_values = h_grad_tensor
                
            # Calculate L2 norm (Euclidean norm)
            # First flatten the tensor to a 1D array
            if hasattr(grad_values, 'flatten'):
                flattened = grad_values.flatten()
                # Calculate L2 norm as sqrt(sum(x_i^2))
                l2_norm = float(jnp.sqrt(jnp.sum(flattened * flattened)))
                vode_grad_norms.append(l2_norm)
                # print(f"Vode {i} L2 norm: {l2_norm}")
            else:
                # If we can't flatten, just log that we couldn't calculate the norm
                print(f"Couldn't calculate norm for vode {i}: grad_values not flattenable")
    
    except Exception as e:
        print(f"Error extracting gradient norms: {e}")
        import traceback
        traceback.print_exc()
        
    # Return the actual calculated norms
    return vode_grad_norms

def create_grouped_bar_chart(table_data, group_col, x_col, y_col, title):
    """
    Create a grouped bar chart for Weights & Biases using custom HTML.
    
    Args:
        table_data: List of rows with data
        group_col: Column name for grouping (e.g., "epoch")
        x_col: Column name for x-axis (e.g., "vode_index")
        y_col: Column name for y-values (e.g., "energy")
        title: Chart title
    
    Returns:
        wandb.Html object with the chart
    """
    # Organize data by group
    groups = {}
    x_values = set()
    
    for row in table_data:
        group = row[0]  # epoch
        x = row[1]      # vode_index
        y = row[2]      # energy/grad_norm
        
        if group not in groups:
            groups[group] = {}
        
        groups[group][x] = y
        x_values.add(x)
    
    # Sort the x values
    x_values = sorted(list(x_values))
    group_keys = sorted(groups.keys())
    
    # Create vega-lite specification
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 500,
        "height": 300,
        "data": {"values": []},
        "mark": "bar",
        "encoding": {
            "x": {"field": x_col, "type": "ordinal", "title": x_col},
            "y": {"field": y_col, "type": "quantitative", "title": y_col},
            "color": {"field": group_col, "type": "nominal", "title": group_col},
            "tooltip": [
                {"field": x_col, "type": "ordinal"},
                {"field": y_col, "type": "quantitative"},
                {"field": group_col, "type": "nominal"}
            ]
        }
    }
    
    # Add data points
    for group in group_keys:
        for x in x_values:
            if x in groups[group]:
                vega_spec["data"]["values"].append({
                    x_col: f"Vode {x}",
                    y_col: groups[group][x],
                    group_col: f"Epoch {group}"
                })
    
    # Create HTML with vega-lite
    html = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
        <div id="vis"></div>
        <script type="text/javascript">
            const spec = {json.dumps(vega_spec)};
            vegaEmbed('#vis', spec);
        </script>
    </body>
    </html>
    """
    
    return wandb.Html(html)

def create_multi_line_chart(table_data, x_col, y_col, series_col, title):
    """
    Create a multi-line chart for Weights & Biases using custom HTML.
    
    Args:
        table_data: List of rows with data
        x_col: Column name for x-axis (e.g., "epoch")
        y_col: Column name for y-values (e.g., "energy")
        series_col: Column name for different lines (e.g., "vode_index")
        title: Chart title
    
    Returns:
        wandb.Html object with the chart
    """
    # Organize data by series
    series_map = {}
    
    for row in table_data:
        x = row[0]      # epoch or inference_step
        series = row[2]  # Corrected: vode_index or series label is the 3rd element
        y = row[1]      # Corrected: energy/grad_norm is the 2nd element
        
        if series not in series_map:
            series_map[series] = []
        
        series_map[series].append({"x": x, "y": y})
    
    # Create vega-lite specification
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 500,
        "height": 300,
        "data": {"values": []},
        "mark": "line",
        "encoding": {
            "x": {"field": x_col, "type": "quantitative", "title": x_col},
            "y": {"field": y_col, "type": "quantitative", "title": y_col},
            "color": {"field": series_col, "type": "nominal", "title": series_col},
            "tooltip": [
                {"field": x_col, "type": "quantitative"},
                {"field": y_col, "type": "quantitative"},
                {"field": series_col, "type": "nominal"}
            ]
        }
    }
    
    # Add data points
    for series, points in series_map.items():
        for point in points:
            vega_spec["data"]["values"].append({
                x_col: point["x"],
                y_col: point["y"],
                series_col: series # Use the actual series label directly
            })
    
    # Create HTML with vega-lite
    html = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
        <div id="vis"></div>
        <script type="text/javascript">
            const spec = {json.dumps(vega_spec)};
            vegaEmbed('#vis', spec);
        </script>
    </body>
    </html>
    """
    
    return wandb.Html(html)

def run_experiment(base_config_name: str = DEFAULT_CONFIG,
                     config_overrides: Optional[Dict[str, Any]] = None,
                     wandb_project_name: str = "debug-transformer-search",
                     wandb_run_name: Optional[str] = None,
                     wandb_mode: str = "online"):
    """Main function to run the debugging process with W&B logging."""
    # args = parse_args() # Args will be handled by overrides or a new main
    
    # Load the base configuration
    if base_config_name not in MODEL_CONFIGS:
        print(f"Error: Base config name '{base_config_name}' not found. Available: {list(MODEL_CONFIGS.keys())}")
        return float('inf') # Return a high MSE indicating failure
        
    config = MODEL_CONFIGS[base_config_name]
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Key '{key}' not found in ModelConfig. Skipping override.")

    # Override with any command-line arguments if needed (though typically covered by config_overrides for programmatic calls)
    # This part is mostly for when this function is adapted to be called from a CLI-driven main()
    # For pure programmatic calls, config_overrides is the primary way.
    # args = parse_args() # We'll handle CLI args in a separate main()
    # for arg_name, arg_value in vars(args).items():
    #     if arg_value is not None and arg_name != 'config': # 'config' here refers to the base_config_name
    #         if hasattr(config, arg_name):
    #             setattr(config, arg_name, arg_value)
    
    # Print the effective configuration
    print(f"\nUsing base configuration '{config.name}' with effective settings:")
    for key, value in vars(config).items():
        if key != 'name': # name is already part of the base config
            print(f"  {key}: {value}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    
    # Determine run name for WandB
    effective_wandb_run_name = wandb_run_name
    if not effective_wandb_run_name:
        # Create a more descriptive run name if not provided
        lr_w_str = f"lrw{config.peak_lr_weights:.0e}" if config.peak_lr_weights else "lrwDEF"
        lr_h_str = f"lrw{config.peak_lr_hidden:.0e}" if config.peak_lr_hidden else "lrhDEF"
        nb_str = f"nb{config.num_blocks}"
        hs_str = f"hs{config.hidden_size}"
        effective_wandb_run_name = f"{config.name}_{nb_str}_{hs_str}_{lr_w_str}_{lr_h_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize Weights & Biases
    run = wandb.init(
        entity="neural-machines", # Replace with your entity or remove if not needed
        project=wandb_project_name,
        name=effective_wandb_run_name,
        config=vars(config),
        mode=wandb_mode  # Allows disabling for search
    )
    
    # Upload code artifacts to W&B if an active run exists and not disabled
    if wandb.run and wandb.run.mode != "disabled":
        try:
            # Use a consistent artifact name, W&B will version it
            code_artifact = wandb.Artifact(name=f"source_code_{config.name}", type="code")
            
            # Add the current file (debug_transformer_wandb.py)
            code_artifact.add_file(__file__)
            
            # Add the decoder_transformer.py file
            # Assuming src is one level up from examples
            script_dir = os.path.dirname(os.path.abspath(__file__))
            decoder_transformer_path = os.path.join(script_dir, "../src/decoder_transformer.py")
            decoder_transformer_path = os.path.abspath(decoder_transformer_path)
            if os.path.exists(decoder_transformer_path):
                code_artifact.add_file(decoder_transformer_path, name="src/decoder_transformer.py") # Explicitly name it for clarity in W&B UI
            else:
                print(f"Warning: Could not find decoder_transformer.py at {decoder_transformer_path}")
            
            wandb.run.log_artifact(code_artifact)
            print("Logged source code artifact to W&B.")
        except Exception as e:
            print(f"Error uploading code to W&B: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Creating configuration for debugging CIFAR-10 transformer...")
    model_config = create_config(
        dataset=config.dataset,
        hidden_size=config.hidden_size,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        patch_size=config.patch_size,
        axes_dim=config.axes_dim,
        theta=config.theta,
        use_noise=config.use_noise,
        use_lower_half_mask=config.use_lower_half_mask,
        use_inference_lr_scaling=config.use_inference_lr_scaling,
        inference_lr_scale_lower=config.inference_lr_scale_lower,
        inference_lr_scale_upper=config.inference_lr_scale_upper,
        inference_lr_scale_boundary=config.inference_lr_scale_boundary,
        inference_clamp_alpha=config.inference_clamp_alpha,
        update_weights_during_unmasking=config.update_weights_during_unmasking,
        use_status_init_in_training=config.use_status_init_in_training,
        use_status_init_in_unmasking=config.use_status_init_in_unmasking,
        update_weights_every_inference_step=config.update_weights_every_inference_step
    )
    
    print(f"Creating debug dataloaders for CIFAR-10...")
    train_loader, val_loader = get_debug_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        root_path=config.data_dir,
        train_subset_n=config.train_subset,
        test_subset_n=config.test_subset,
        target_class=config.target_class
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    print("Initializing model...")
    model = TransformerDecoder(model_config)
    
    # Calculate steps per epoch for learning rate schedule
    steps_per_epoch = len(train_loader)
    
    # Create learning rate functions based on config
    if config.use_lr_schedule:
        # Use schedule with warmup and decay
        weights_lr_fn = create_learning_rate_schedule(
            config.peak_lr_weights, 
            config.warmup_epochs, 
            config.epochs, 
            steps_per_epoch
        )
        
        hidden_lr_fn = create_learning_rate_schedule(
            config.peak_lr_hidden, 
            config.warmup_epochs, 
            config.epochs, 
            steps_per_epoch
        )
        print(f"Using learning rate schedule - Peak weights LR: {config.peak_lr_weights}, Peak hidden LR: {config.peak_lr_hidden}")
    else:
        # Use constant learning rates
        weights_lr_fn = config.peak_lr_weights
        hidden_lr_fn = config.peak_lr_hidden
        print(f"Using constant learning rates - Weights: {weights_lr_fn}, Hidden: {hidden_lr_fn}")
    
    # Create optimizers with the appropriate learning rate function
    # TODO: add gradient clipping
    optim_h = pxu.Optim(lambda: optax.sgd(hidden_lr_fn, momentum=0.1))
    optim_h_inference = pxu.Optim(lambda: optax.sgd(config.hidden_lr_inference, momentum=0.1))
    optim_w = pxu.Optim(
        lambda: optax.adamw(weights_lr_fn, weight_decay=config.weight_decay), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    # Store validation losses
    # val_losses = [] # Replaced by metrics dict
    
    # Store energy and gradient data for all epochs
    all_epochs_energy_data = []
    all_epochs_grad_data = []
    
    # Early stopping variables
    best_val_loss = float('inf') # Will track the primary metric (e.g., masked_mse)
    # val_loss = float('inf') # Replaced by pretext_metrics dictionary
    epochs_without_improvement = 0
    early_stopped = False
    early_stopped_epoch = -1 # Store epoch when stopped
    pretext_metrics = {} # Initialize metrics dict
    
    print(f"Training for {config.epochs} epochs with W&B logging...")

    # Initialize the model (set h values of the Vodes) using a dummy batch shape
    # Determine expected input shape: (batch_size, channels, height, width)
    init_shape = (config.batch_size, *model_config.image_shape)
    x_init = jnp.zeros(init_shape, dtype=jnp.float32) # Use float32 or model's dtype
    print(f"Initializing Vode states using dummy tensor with shape: {init_shape}")
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x_init, model=model) # Use dummy tensor for shape initialization
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # Get current learning rates - handle both scheduled and constant LR cases
        if config.use_lr_schedule:
            current_w_lr = weights_lr_fn(epoch * steps_per_epoch)
            current_h_lr = hidden_lr_fn(epoch * steps_per_epoch)
        else:
            current_w_lr = weights_lr_fn
            current_h_lr = hidden_lr_fn
            
        print(f"Current learning rates - Weights: {current_w_lr:.6f}, Hidden: {current_h_lr:.6f}")

        if config.reinitialize_model_for_each_epoch:
            # TODO: Not sure about its usage. It could be causing instability. It could make representations more robust.
            print(f"Reinitializing model for epoch {epoch+1}...")
            with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                forward(x_init, model=model)
        
        # Train for one epoch
        avg_train_w_energy, avg_train_mse, h_grad, w_grad = train(
            train_loader, config.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch
        )
        
        # Initialize epoch_metrics here to store training metrics
        epoch_metrics = {
            'Losses/train_w_energy_avg': avg_train_w_energy,
            'Losses/train_mse_avg': avg_train_mse,
            'LearningRate/weights': current_w_lr,
            'LearningRate/hidden': current_h_lr
        }

        if (epoch + 1) % config.validation_every_n_epochs == 0 or epoch == config.epochs - 1 or (config.use_early_stopping and early_stopped and epoch == early_stopped_epoch):
            print(f"Evaluating pretext task metrics on validation set...")
            # Use config settings for evaluation masking
            # NOTE: T_values for eval might differ from training T
            # Use val_loader here
            pretext_metrics = eval_pretext_metrics(
                val_loader, # Use validation loader
                T_values=config.eval_inference_steps, # Use eval_inference_steps
                use_corruption=config.use_corruption, 
                corrupt_ratio=config.corrupt_ratio,
                model=model, 
                optim_h=optim_h,
                optim_w=optim_w # <<< Pass optim_w
            )
            print(f"Validation Pretext Metrics: {pretext_metrics}")
            
            # Add validation metrics with clear names
            for key, value in pretext_metrics.items():
                epoch_metrics[f'ValMetrics/Pretext/{key}'] = value

            # Log detailed vode statistics and get processed data for summary
            # Pass h_grad, w_grad from the end of the training epoch
            if h_grad is not None and w_grad is not None:
                processed_energy_data, processed_grad_data = log_vode_stats(model, h_grad, w_grad, run, epoch)
                # Add the processed data to our summary collections
                all_epochs_energy_data.extend(processed_energy_data)
                all_epochs_grad_data.extend(processed_grad_data)
            else:
                print("Warning: Gradients not available for Vode stats logging.")

            # Initialize best_val_loss on the first validation run
            current_val_metric = pretext_metrics.get('masked_mse', float('inf')) # Use masked_mse for early stopping
            if best_val_loss == float('inf') and current_val_metric != float('inf'):
                 best_val_loss = current_val_metric
                 print(f"Initialized best_val_loss for early stopping with masked_mse: {best_val_loss:.6f}")
            
             # Early stopping check based on the chosen metric (e.g., masked_mse)
            if config.use_early_stopping and best_val_loss != float('inf'): # Ensure best_val_loss is initialized
                if current_val_metric < (best_val_loss - config.early_stopping_min_delta):
                    print(f"Validation metric improved! {best_val_loss:.6f} -> {current_val_metric:.6f}")
                    best_val_loss = current_val_metric
                    epochs_without_improvement = 0
                    # Save the best model here if desired
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement in validation metric for {epochs_without_improvement} epochs (best: {best_val_loss:.6f})")
                    if epochs_without_improvement >= config.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs!")
                        early_stopped = True
                        early_stopped_epoch = epoch
                        # Log final metrics before breaking
                        run.log(epoch_metrics, step=epoch+1)
                        break # Exit training loop
        
        # Log metrics collected so far for this epoch (includes train loss, potentially val metrics)
        run.log(epoch_metrics, step=epoch+1)  # Use epoch+1 to start from step 1

        # Generate reconstructions every N epochs (and for the final epoch)
        # Base the condition on one of the new validation metrics if available and valid
        # Use the most recently calculated validation metric if available
        current_val_metric_for_recon = pretext_metrics.get('masked_mse', float('inf')) 

        # Trigger reconstruction based on interval OR significant improvement OR final epoch/stop
        trigger_reconstruction = False
        if (epoch + 1) % config.reconstruction_every_n_epochs == 0 and current_val_metric_for_recon < float('inf'):
            trigger_reconstruction = True
        elif best_val_loss != float('inf') and current_val_metric_for_recon < best_val_loss - 0.01: # Use same threshold as early stopping improvement check
            trigger_reconstruction = True
        elif epoch == config.epochs - 1: # Final epoch
            trigger_reconstruction = True
        elif early_stopped and epoch == early_stopped_epoch: # After early stopping
             trigger_reconstruction = True
             
        if trigger_reconstruction:
            print(f"Generating reconstructions for epoch {epoch+1} (Val Metric: {current_val_metric_for_recon:.6f})...")
            # Pass the config object here
            # Use train_loader for visualizing training samples
            vis_path, vis_logs = visualize_reconstruction(
                model, 
                model_config, 
                optim_h_inference, 
                train_loader, 
                config=config, 
                wandb_run=run,
                epoch=epoch+1,
                optim_w=optim_w # <<< Pass optim_w
            )
            # Add visualization logs (either image or video) to wandb logging for this step
            run.log(vis_logs, step=epoch+1) 
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")
        # Break loop if early stopping occurred in the validation check
        if early_stopped and epoch == early_stopped_epoch:
            break
    
    # Create summary tables after training is complete
    if all_epochs_energy_data:
        summary_energy_table = wandb.Table(data=all_epochs_energy_data, 
                                          columns=["epoch", "vode_index", "energy"])
        run.log({
            "Summary/VodeEnergy_AllEpochs": create_grouped_bar_chart(all_epochs_energy_data, "epoch", "vode_index", "energy", "Energy by Vode Across All Epochs")
        })
        
        # Also create a line plot to show the trend over epochs
        run.log({
            "Summary/VodeEnergy_Trends": create_multi_line_chart(all_epochs_energy_data, "epoch", "energy", "vode_index", "Energy Trends by Vode Across Epochs")
        })
    
    if all_epochs_grad_data:
        summary_grad_table = wandb.Table(data=all_epochs_grad_data, 
                                        columns=["epoch", "vode_index", "grad_norm"])
        run.log({
            "Summary/VodeGradNorm_AllEpochs": create_grouped_bar_chart(all_epochs_grad_data, "epoch", "vode_index", "grad_norm", "Gradient Norm by Vode Across All Epochs")
        })
        
        # Also create a line plot to show the trend over epochs
        run.log({
            "Summary/VodeGradNorm_Trends": create_multi_line_chart(all_epochs_grad_data, "epoch", "grad_norm", "vode_index", "Gradient Norm Trends by Vode Across Epochs")
        })
        
    # Close W&B after all logging is complete
    wandb.finish()
    
    print("Debug run completed!")
    # Return the average training MSE of the last epoch
    # Assuming avg_train_mse is updated each epoch and we want the one from the final epoch of training.
    # If early stopping happens, this will be the MSE from the epoch it stopped.
    last_train_mse = epoch_metrics.get('Losses/train_mse_avg', float('inf')) # Get from last logged metrics
    
    # A more robust way if epoch_metrics might not be populated if training is very short (e.g. 0 epochs)
    # or if the loop was broken before epoch_metrics was set.
    # For simplicity, we assume epochs > 0 and epoch_metrics is set.
    # If training loop doesn't run, avg_train_mse might not be defined.
    # Let's ensure we return a value from within the training loop if possible or a default.
    final_avg_train_mse = float('inf')
    if 'avg_train_mse' in locals() or 'avg_train_mse' in globals():
         final_avg_train_mse = avg_train_mse # This would be from the last completed epoch
    elif 'epoch_metrics' in locals() and 'Losses/train_mse_avg' in epoch_metrics:
         final_avg_train_mse = epoch_metrics['Losses/train_mse_avg']
    
    print(f"Run {effective_wandb_run_name} finished with final avg_train_mse: {final_avg_train_mse}")
    return final_avg_train_mse

if __name__ == "__main__":
    cli_args = parse_args()
    
    # Prepare config_overrides from CLI arguments, excluding 'config' which is used for base_config_name
    overrides = {k: v for k, v in vars(cli_args).items() if v is not None and k != 'config'}
    
    run_experiment(base_config_name=cli_args.config, 
                   config_overrides=overrides) 