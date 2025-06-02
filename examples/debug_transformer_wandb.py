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
import jax.random as jrand
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
import equinox as eqx

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

from src.utils import create_grouped_bar_chart, create_multi_line_chart

# Import the pcx RandomKeyGenerator
from pcx.core._random import RKG # Assuming this is the correct import path for RKG
import random # Import Python's random module
from examples.linear_probe import run_linear_probe_evaluation

from src.config import MODEL_CONFIGS, ModelConfig, DEFAULT_CONFIG, create_config

try:
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Kornia not available. Install with: pip install kornia")


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
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Hidden size of transformer')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--peak_lr_weights', type=float, default=None,
                        help='Peak learning rate for weights')
    parser.add_argument('--peak_lr_hidden', type=float, default=None,
                        help='Peak learning rate for hidden states')
    parser.add_argument('--inference_lr_scale_base', type=float, default=None,
                        help='Base scaling factor for inference learning rates')
    parser.add_argument('--hidden_momentum', type=float, default=None,
                        help='Momentum for hidden state optimizer')
    parser.add_argument('--h_grad_clip_norm', type=float, default=None,
                        help='Gradient clipping norm for hidden states')
    parser.add_argument('--w_grad_clip_norm', type=float, default=None,
                        help='Gradient clipping norm for weights')
    parser.add_argument('--inference_steps', type=int, default=None,
                        help='Number of inference steps')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of warmup steps for learning rate schedule')
    parser.add_argument('--use_ssl_augmentations', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Use SSL data augmentations (true/false)')
    parser.add_argument('--use_cifar10_norm', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Use CIFAR-10 specific normalization (true/false)')
    parser.add_argument('--use_lr_schedule_h', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Use learning rate schedule for hidden states (true/false)')
    parser.add_argument('--use_lr_schedule_w', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Use learning rate schedule for weights (true/false)')
    parser.add_argument('--corrupt_ratio', type=float, default=None,
                        help='Ratio of corruption for masking')
    parser.add_argument('--use_lower_half_mask', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Use lower half masking strategy (true/false)')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience')
    parser.add_argument('--linear_probe_every_n_epochs', type=int, default=None,
                        help='Run linear probe every N epochs')
    parser.add_argument('--linear_probe_epochs', type=int, default=None,
                        help='Number of epochs for linear probe training')
    parser.add_argument('--linear_probe_lr', type=float, default=None,
                        help='Learning rate for linear probe')
    parser.add_argument('--linear_probe_wd', type=float, default=None,
                        help='Weight decay for linear probe')
    parser.add_argument('--test_subset', type=int, default=None,
                        help='Size of test subset')
    parser.add_argument('--train_subset', type=int, default=None,
                        help='Size of training subset')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images for visualization')
    parser.add_argument('--save_reconstruction_images', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Save reconstruction images (true/false)')
    parser.add_argument('--save_reconstruction_video', type=str_to_bool, nargs='?', const=True, default=None,
                        help='Save reconstruction video (true/false)')
    parser.add_argument('--intermediate_l1_coeff', type=float, default=None,
                        help='L1 regularization coefficient for intermediate layers')
    parser.add_argument('--intermediate_l2_coeff', type=float, default=None,
                        help='L2 regularization coefficient for intermediate layers')
    parser.add_argument('--sweep', action='store_true',
                        help='Run as part of a wandb sweep (gets config from wandb.config)')
    # Add any other parameters from ModelConfig you want to control via CLI here
    return parser.parse_args()


def create_learning_rate_schedule(base_lr, warmup_steps, total_steps, min_lr_factor):
    """Create a learning rate schedule with warmup and cosine decay to a factor of the peak rate."""
    min_lr = base_lr * min_lr_factor
    
    def lr_schedule(step):
        # Ensure step is an integer or float for calculations
        current_step = jnp.float32(step)

        # Calculate arguments for decay phase
        # Effective start of decay phase in terms of steps
        decay_begins_at_step = jnp.float32(warmup_steps)
        # Duration of the decay phase in steps
        decay_phase_duration_steps = jnp.float32(total_steps - warmup_steps)
        
        # Ensure decay_phase_duration_steps is at least a small positive number
        safe_decay_phase_duration_steps = jnp.maximum(decay_phase_duration_steps, 1e-8)

        # Progress into the decay phase (can be negative if still in warmup)
        progress_into_decay_steps = current_step - decay_begins_at_step
        
        decay_ratio = progress_into_decay_steps / safe_decay_phase_duration_steps
        decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0) 
        
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        decay_lr_value = min_lr + (base_lr - min_lr) * cosine_factor

        if warmup_steps > 0:
            # Calculate linear warmup learning rate
            # Progress through warmup phase, clipped to [0,1]
            warmup_phase_progress = jnp.clip(current_step / jnp.float32(warmup_steps), 0.0, 1.0)
            warmup_lr_value = base_lr * warmup_phase_progress
            
            # Select LR based on whether current_step is in warmup or decay phase
            current_lr = jnp.where(current_step < decay_begins_at_step, warmup_lr_value, decay_lr_value)
        else:
            # No warmup steps, so current_lr is directly the decay_lr_value
            current_lr = decay_lr_value
            
        return current_lr
            
    return lr_schedule


def get_debug_dataloaders(dataset_name, batch_size, root_path, train_subset_n=None, test_subset_n=None, target_class=None, use_ssl_augmentations=True, use_cifar10_norm=True):
    """Get data loaders with simple augmentation for debugging."""
    
    # Define a common normalization (from vision_transformer_script.py)
    if use_cifar10_norm:
        print("Using CIFAR-10 specific normalization.")
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        print("Using generic (0.5, 0.5, 0.5) normalization.")
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    # Image dimensions (assuming 32x32 for CIFAR-10)
    height, width = 32, 32

    if use_ssl_augmentations: # For now, this flag will enable the ViT script's augmentations
        print("Using minimal CPU augmentations - heavy augmentations will be done on GPU.")
        train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) # Use selected normalization
        ]
    else:
        print("Using basic ToTensor and Normalize for training (CIFAR-10 specific).")
        train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) # Use selected normalization
        ]
    
    train_transform = transforms.Compose(train_transform_list)
    
    test_transform = transforms.Compose([
        transforms.Resize((height, width)), # From vision_transformer_script.py
        transforms.ToTensor(),
        transforms.Normalize(mean, std) # Use selected normalization
    ])
    
    # Fix dataset path to work regardless of where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from examples/ to project root, then add datasets path
    project_root = os.path.dirname(script_dir)
    dataset_root = os.path.join(project_root, "datasets", "cifar10")
    print(f"Using dataset root: {dataset_root}")
    
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
        shuffle=True,
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


def setup_gpu_augmentations(use_ssl_augmentations=True):
    """Setup GPU-accelerated augmentations using Kornia."""
    if not use_ssl_augmentations or not KORNIA_AVAILABLE:
        return None
    
    # GPU augmentations using Kornia
    gpu_augmentations = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # Note: Kornia's GaussianBlur is much faster on GPU if needed
        # K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),
        same_on_batch=False
    )
    return gpu_augmentations


def apply_gpu_augmentations(batch_tensor, gpu_augmentations):
    """Apply GPU augmentations to a batch of images."""
    if gpu_augmentations is None:
        return batch_tensor
    
    # Ensure tensor is on GPU and in float format
    if not batch_tensor.is_cuda:
        batch_tensor = batch_tensor.cuda()
    
    # Apply augmentations
    with torch.no_grad():  # Don't track gradients for augmentations
        augmented = gpu_augmentations(batch_tensor)
    
    return augmented


def apply_cpu_fallback_augmentations(x):
    """Apply CPU-based augmentations when GPU augmentations fail."""
    import torch
    import torchvision.transforms as transforms
    
    # Convert to PyTorch tensor if needed
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.from_numpy(x.numpy()) if hasattr(x, 'numpy') else torch.tensor(x)
    else:
        x_tensor = x
    
    # Ensure tensor is float
    x_tensor = x_tensor.float()
    
    # Define CPU augmentations (same as GPU but using torchvision)
    cpu_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ])
    
    # Apply augmentations batch-wise
    augmented_batch = []
    for i in range(x_tensor.shape[0]):
        # Convert single image from [C, H, W] to PIL format for transforms
        img = x_tensor[i]  # Shape: [C, H, W]
        # torchvision transforms expect [C, H, W] tensor, which we have
        augmented_img = cpu_augmentations(img)
        augmented_batch.append(augmented_img)
    
    # Stack back into batch
    augmented_tensor = torch.stack(augmented_batch)
    
    # Convert back to numpy for JAX
    return augmented_tensor.numpy()


def create_reconstruction_images(intermediate_recons, T_values, orig_images, masked_images, labels_list, num_images, image_shape, wandb_run, epoch, reconstruction_mses=None):
    """
    Creates a grid of images comparing originals, masked inputs, and reconstructions at specific T_values.
    Adds MSE to the title of the final reconstruction if provided.
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
        current_mse_str = f" (MSE: {reconstruction_mses[i]:.4f})" if reconstruction_mses and i < len(reconstruction_mses) else ""
        if num_channels == 1:
            axes[i, 1].imshow(np.clip(np.squeeze(masked_np), 0.0, 1.0), cmap='gray')
        else:
            axes[i, 1].imshow(np.clip(np.transpose(masked_np, (1, 2, 0)), 0.0, 1.0))
        axes[i, 1].set_title(f'Masked Input{current_mse_str}')
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
    image_dir = "../results/images" # New directory
    os.makedirs(image_dir, exist_ok=True) # Ensure directory exists
    reconstruction_path = f"{image_dir}/reconstruction_images{epoch_str}_{timestamp}.png" # Updated path
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
    # num_images = config.num_images # This is the requested number
    requested_num_images = config.num_images # Rename for clarity
    use_corruption = config.use_corruption
    corrupt_ratio = config.corrupt_ratio
    T_values = config.reconstruction_steps # Use T_values from config

    orig_images = []
    recon_images_at_T = {T: [] for T in T_values} # For static images
    all_reconstruction_frames = [] # To store frames for video
    labels_list = []
    masked_images_list = [] # List to store the masked input images
    all_inference_grads = [] # Add list to store grad logs per image
    reconstruction_mses = [] # New: Store MSE for each reconstruction
    dataloader_iter = iter(dataloader)

    actual_num_images_visualized = 0 # Counter for images actually visualized

    # Get the single batch
    try:
        x_batch, label_batch = next(dataloader_iter) # x_batch.shape[0] is batch_size
        x_batch = jnp.array(x_batch)

        # Determine how many images to actually visualize from this single batch
        num_to_visualize_from_this_batch = min(requested_num_images, x_batch.shape[0])

        # Process each image in the batch separately
        for i in range(num_to_visualize_from_this_batch): # Modified loop range
            actual_num_images_visualized += 1
            single_x = x_batch[i:i+1] # This is now safe
            orig_images.append(jnp.reshape(single_x[0], image_shape))
            
            if hasattr(label_batch, 'item'): # Use label_batch here
                labels_list.append(label_batch[i].item() if len(label_batch.shape) > 0 else label_batch.item())
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
            reconstruction_mses.append(loss) # Store the MSE for this image

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
        print("Warning: No data available in dataloader for visualization.")
        # actual_num_images_visualized will remain 0
    
    # --- Create Visualizations based on config ---
    final_path = None
    combined_log_dict = {}

    if actual_num_images_visualized == 0: # Check if any images were processed
        print("No images were visualized from the batch.")
        return None, {}


    # --- Process and log inference gradients ---
    if all_inference_grads: # This list will have actual_num_images_visualized elements
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
            num_images=actual_num_images_visualized, # Pass the actual number processed
            image_shape=image_shape,
            wandb_run=wandb_run,
            epoch=epoch,
            reconstruction_mses=reconstruction_mses
        )
        final_path = image_path # Prioritize image path if both are created
        combined_log_dict.update(image_log_dict)

    # Create video if requested
    if config.save_reconstruction_video:
        if not all_reconstruction_frames: # This list will have actual_num_images_visualized items
            print("Warning: No reconstruction frames generated for video.")
        else:
            video_path, video_log_dict = create_reconstruction_video(
                all_reconstruction_frames=all_reconstruction_frames,
                orig_images=orig_images,
                masked_images=masked_images_list, # Pass masked images
                labels_list=labels_list,
                num_images=actual_num_images_visualized, # Pass the actual number processed
                image_shape=image_shape,
                wandb_run=wandb_run,
                epoch=epoch,
                fps=config.video_fps, # Use fps from config
                reconstruction_mses=reconstruction_mses, # Pass MSEs
            )
            if final_path is None:
                final_path = video_path # Set video path if images weren't created
            combined_log_dict.update(video_log_dict)

    return final_path, combined_log_dict


def create_reconstruction_video(all_reconstruction_frames, orig_images, masked_images, labels_list, num_images, image_shape, wandb_run, epoch, fps=10, reconstruction_mses=None):
    """
    Creates a video comparing original images and their reconstructions over time.
    Adds MSE to the title of the reconstruction if provided.

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
        reconstruction_mses: List of MSE values for each image.

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
            current_mse_str = f" (MSE: {reconstruction_mses[i]:.4f})" if reconstruction_mses and i < len(reconstruction_mses) else ""

            if num_channels == 1: # Grayscale
                recon_plot = np.clip(np.squeeze(recon_np), 0.0, 1.0)
                recon_plot = (plt.cm.gray(recon_plot)[:, :, :3] * 255).astype(np.uint8)
            else: # RGB
                recon_plot = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)
                recon_plot = (recon_plot * 255).astype(np.uint8)
            
            # Plot in column 2
            axes[i, 2].imshow(recon_plot)
            axes[i, 2].set_title(f'Recon T={t+1}{current_mse_str}')
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
    video_dir = "../results/videos" # New directory
    os.makedirs(video_dir, exist_ok=True) # Ensure directory exists
    video_path = f"{video_dir}/reconstruction_video{epoch_str}_{timestamp}.mp4" # Updated path, save as MP4
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


def run_experiment(base_config_name: str = DEFAULT_CONFIG,
                     config_overrides: Optional[Dict[str, Any]] = None,
                     wandb_project_name: str = "debug-transformer-search",
                     wandb_run_name: Optional[str] = None,
                     wandb_mode: str = "online",
                     wandb_run: Optional[Any] = None):
    """Main function to run the debugging process with W&B logging."""
    
    # --- Seed everything FIRST for reproducibility ---
    if config_overrides and "seed" in config_overrides:
        current_seed = config_overrides["seed"]
    elif hasattr(config, 'seed'):
        current_seed = config.seed
    else:
        current_seed = 42 # Fallback seed if not found
        print(f"Warning: Seed not found in config or overrides, using fallback: {current_seed}")

    print(f"Using master seed: {current_seed}")
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    RKG.seed(current_seed) # Re-seed the pcx global RandomKeyGenerator
    
    # Master JAX key for this experiment run - operations in run_experiment will derive from this
    master_key = jax.random.PRNGKey(current_seed) 
    best_linear_probe_accuracy_overall = 0.0 # Initialize best probe accuracy
    # --- End Seeding ---
    
    # Load the base configuration
    if base_config_name not in MODEL_CONFIGS:
        print(f"Error: Base config name '{base_config_name}' not found. Available: {list(MODEL_CONFIGS.keys())}")
        # Return structure for hyperparam_search: best_val_mse, run_best_train_mse, final_train_mse, early_stop_reason
        return float('inf'), float('inf'), float('inf'), "ConfigNotFound" 
        
    config = MODEL_CONFIGS[base_config_name]
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Key '{key}' not found in ModelConfig. Skipping override.")

    
    # Print the effective configuration
    print(f"\nUsing base configuration '{config.name}' with effective settings:")
    for key, value in vars(config).items():
        if key != 'name': # name is already part of the base config
            print(f"  {key}: {value}")
    print()
    
    
    # Determine run name for WandB
    effective_wandb_run_name = wandb_run_name
    if not effective_wandb_run_name:
        # Create a more descriptive run name if not provided
        lr_w_str = f"lrw{config.peak_lr_weights:.0e}" if config.peak_lr_weights else "lrwDEF"
        lr_h_str = f"lrh{config.peak_lr_hidden:.2f}".replace('.', 'p') # Format for clarity
        nb_str = f"nb{config.num_blocks}"
        hs_str = f"hs{config.hidden_size}"
        scale_base_str = f"sbase{config.inference_lr_scale_base:.2f}".replace('.', 'p') if config.use_inference_lr_scaling and config.inference_lr_scale_base is not None else "sbaseOFF"
        hclip_str = f"hclip{config.h_grad_clip_norm}" if config.h_grad_clip_norm is not None else "hclipOFF"
        wclip_str = f"wclip{config.w_grad_clip_norm}" if config.w_grad_clip_norm is not None else "wclipOFF"
        
        effective_wandb_run_name = f"{config.name}_{nb_str}_{hs_str}_{lr_w_str}_{lr_h_str}_{scale_base_str}_{hclip_str}_{wclip_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize Weights & Biases (only if not provided)
    if wandb_run is None:
        run = wandb.init(
            entity="neural-machines", # Replace with your entity or remove if not needed
            project=wandb_project_name,
            name=effective_wandb_run_name,
            config=vars(config),
            mode=wandb_mode  # Allows disabling for search
        )
    else:
        run = wandb_run
        # Update the config in the existing run
        wandb.config.update(vars(config))
    
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
            
            # Add the hyperparam_search.py file
            hyperparam_search_path = os.path.join(script_dir, "hyperparam_search.py")
            hyperparam_search_path = os.path.abspath(hyperparam_search_path)
            if os.path.exists(hyperparam_search_path):
                code_artifact.add_file(hyperparam_search_path, name="examples/hyperparam_search.py") # Explicitly name it
            else:
                print(f"Warning: Could not find hyperparam_search.py at {hyperparam_search_path}")
            
            wandb.run.log_artifact(code_artifact)
            print("Logged source code artifact to W&B.")
        except Exception as e:
            print(f"Error uploading code to W&B: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Creating configuration for CIFAR-10 transformer...")
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
        inference_lr_scale_base=config.inference_lr_scale_base,
        inference_clamp_alpha=config.inference_clamp_alpha,
        update_weights_during_unmasking=config.update_weights_during_unmasking,
        use_status_init_in_training=config.use_status_init_in_training,
        use_status_init_in_unmasking=config.use_status_init_in_unmasking,
        update_weights_every_inference_step=config.update_weights_every_inference_step,
        use_vode_state_layernorm=config.use_vode_state_layernorm, # New
        use_vode_grad_norm=config.use_vode_grad_norm,             # New
        vode_grad_norm_target=config.vode_grad_norm_target,        # New
        intermediate_l1_coeff=config.intermediate_l1_coeff, # ADDED
        intermediate_l2_coeff=config.intermediate_l2_coeff  # ADDED
    )
    
    print(f"Creating debug dataloaders for CIFAR-10...")
    train_loader, val_loader = get_debug_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        root_path=config.data_dir,
        train_subset_n=config.train_subset,
        test_subset_n=config.test_subset,
        target_class=config.target_class,
        use_ssl_augmentations=config.use_ssl_augmentations, # Pass the flag
        use_cifar10_norm=config.use_cifar10_norm # Pass the new flag
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    print("Initializing model...")
    model = TransformerDecoder(model_config)
    
    # Calculate steps per epoch for learning rate schedule
    steps_per_epoch = len(train_loader)
    total_train_steps = config.epochs * steps_per_epoch # Total steps for the entire training
    
    # Create learning rate functions based on config
    if config.use_lr_schedule_w:
        weights_lr_fn = create_learning_rate_schedule(
            config.peak_lr_weights, 
            config.warmup_steps,
            total_train_steps,
            config.lr_schedule_min_lr_factor
        )
        print(f"Using LR schedule for weights: Peak {config.peak_lr_weights}, Warmup {config.warmup_steps}, Total {total_train_steps}, MinFactor {config.lr_schedule_min_lr_factor}")
    else:
        weights_lr_fn = config.peak_lr_weights # Fixed LR for weights
        print(f"Using FIXED learning rate for weights: {weights_lr_fn:.0e}")
        
    if config.use_lr_schedule_h:
        hidden_lr_fn = create_learning_rate_schedule(
            config.peak_lr_hidden, 
            config.warmup_steps,  # Assuming shared warmup steps for now
            total_train_steps,
            config.lr_schedule_min_lr_factor # Assuming shared min factor
        )
        print(f"Using LR schedule for hidden: Peak {config.peak_lr_hidden}, Warmup {config.warmup_steps}, Total {total_train_steps}, MinFactor {config.lr_schedule_min_lr_factor}")
    else:
        hidden_lr_fn = config.peak_lr_hidden # Fixed LR for hidden
        print(f"Using FIXED learning rate for hidden: {hidden_lr_fn}")
    
    # --- Create base optimizers --- 
    if config.use_adamw_for_hidden_optimizer:
        print("Using AdamW for hidden state optimizer.")
        # AdamW for hidden states typically does not use weight decay.
        # Learning rate is provided by hidden_lr_fn (for training) or config.hidden_lr_inference (for inference).
        # Standard beta values for AdamW are b1=0.9, b2=0.999. Epsilon is 1e-8.
        base_optim_h_train = optax.adamw(learning_rate=hidden_lr_fn, b1=0.9, b2=0.999, eps=1e-8)
        base_optim_h_inference = optax.adamw(learning_rate=config.hidden_lr_inference, b1=0.9, b2=0.999, eps=1e-8)
    else:
        print("Using SGD for hidden state optimizer.")
    base_optim_h_train = optax.sgd(hidden_lr_fn, momentum=config.hidden_momentum)
    base_optim_h_inference = optax.sgd(config.hidden_lr_inference, momentum=config.hidden_momentum)
    
    base_optim_w = optax.adamw(weights_lr_fn, weight_decay=config.weight_decay)
    
    # --- Apply gradient clipping if configured ---

    # Clipping for H gradients (training)
    optim_h_train_steps = []
    if config.h_grad_clip_norm is not None and config.h_grad_clip_norm > 0:
        print(f"Applying H-gradient clipping with max_norm = {config.h_grad_clip_norm}")
        h_clipper = optax.clip_by_global_norm(config.h_grad_clip_norm)
        optim_h_train_steps.append(h_clipper)
    optim_h_train_steps.append(base_optim_h_train)
    final_optim_h_train = optax.chain(*optim_h_train_steps)

    # Clipping for H gradients (inference)
    optim_h_inference_steps = []
    if config.h_grad_clip_norm is not None and config.h_grad_clip_norm > 0:
        # Assuming same clipping for training and inference h grads, if not, add another config field
        if 'h_clipper' not in locals(): # Define h_clipper if not already defined (e.g. if h_grad_clip_norm was only for training)
            h_clipper = optax.clip_by_global_norm(config.h_grad_clip_norm)
        optim_h_inference_steps.append(h_clipper)
    optim_h_inference_steps.append(base_optim_h_inference)
    final_optim_h_inference = optax.chain(*optim_h_inference_steps)

    # Clipping for W gradients
    optim_w_steps = []
    if config.w_grad_clip_norm is not None and config.w_grad_clip_norm > 0:
        print(f"Applying W-gradient clipping with max_norm = {config.w_grad_clip_norm}")
        w_clipper = optax.clip_by_global_norm(config.w_grad_clip_norm)
        optim_w_steps.append(w_clipper)
    optim_w_steps.append(base_optim_w)
    final_optim_w = optax.chain(*optim_w_steps)

    # --- Create pcx Optim wrappers using the final optimizers ---
    optim_h = pxu.Optim(lambda: final_optim_h_train)
    optim_h_inference = pxu.Optim(lambda: final_optim_h_inference)
    optim_w = pxu.Optim(
        lambda: final_optim_w, 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    # Store validation losses
    # val_losses = [] # Replaced by metrics dict
    
    # Store energy and gradient data for all epochs
    all_epochs_energy_data = []
    all_epochs_grad_data = []
    
    # Early stopping & Model saving variables
    best_val_loss_for_overall_best_model = float('inf') # Tracks best validation loss overall for a dedicated save
    best_metric_for_early_stopping = float('inf') # Tracks the chosen metric for the stopping decision
    best_metric_for_saving = float('inf') # Tracks the chosen metric for the primary saving strategy
    can_save_model_now = False # Flag to enable saving after train MSE threshold is met
    epochs_without_improvement_early_stopping = 0 # Counter for early stopping
    epochs_without_improvement_saving = 0 # Counter for saving (not strictly needed if saving on any improvement)

    early_stopped = False
    early_stop_reason = None # Initialize early_stop_reason
    pretext_metrics = {} # Initialize metrics dict
    
    print(f"Training for {config.epochs} epochs with W&B logging...")

    # Setup GPU augmentations if enabled
    gpu_augmentations = setup_gpu_augmentations(config.use_ssl_augmentations)
    if gpu_augmentations is not None and KORNIA_AVAILABLE:
        print("GPU augmentations enabled using Kornia.")
    elif config.use_ssl_augmentations and not KORNIA_AVAILABLE:
        print("Warning: SSL augmentations requested but Kornia not available. Using simple CPU augmentations.")
    else:
        print("No augmentations will be applied.")

    # Initialize best_train_mse_this_run to track the minimum training MSE for the current run
    best_train_mse_this_run = float('inf')

    # Initialize the model (set h values of the Vodes) using a dummy batch shape
    # Determine expected input shape: (batch_size, channels, height, width)
    init_shape = (config.batch_size, *model_config.image_shape)
    x_init = jnp.zeros(init_shape, dtype=jnp.float32) # Use float32 or model's dtype
    print(f"Initializing Vode states using dummy tensor with shape: {init_shape}")

    model_init_key, main_train_key = jax.random.split(master_key) # Split master key

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache): # Removed key=model_init_key
        forward(x_init, model=model)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        new_train_mse_milestone_reached_this_epoch = False # New flag for triggering val/recon
        training_nan_detected = False # Initialize here

        if model_config.use_inference_lr_scaling:
            print(f"Current inference_lr_scale_base: {model_config.inference_lr_scale_base}")
        else:
            print("Inference LR scaling is disabled.")
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # Get current learning rates - handle both scheduled and constant LR cases
        current_global_step = epoch * steps_per_epoch # Calculate once here
        
        if config.use_lr_schedule_w:
            current_w_lr = weights_lr_fn(current_global_step)
        else:
            current_w_lr = weights_lr_fn # This is just the fixed LR value
            
        if config.use_lr_schedule_h:
            current_h_lr = hidden_lr_fn(current_global_step)
        else:
            current_h_lr = hidden_lr_fn # This is just the fixed LR value
            
        print(f"Current learning rates - Weights: {current_w_lr:.6f}, Hidden: {current_h_lr:.6f}")

        if config.reinitialize_model_for_each_epoch:
            # TODO: Not sure about its usage. It could be causing instability. It could make representations more robust.
            print(f"Reinitializing model for epoch {epoch+1}...")
            with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                forward(x_init, model=model)
        
        # Train for one epoch
        avg_train_w_energy, avg_train_mse, h_grad, w_grad = train(
            train_loader, config.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, gpu_augmentations=gpu_augmentations
        )
        
        avg_train_w_energy = float(avg_train_w_energy) if not jnp.isnan(avg_train_w_energy) else float('nan')
        avg_train_mse = float(avg_train_mse) if not jnp.isnan(avg_train_mse) else float('nan')
        
        # Check if train MSE is below threshold for enabling model saving
        if not can_save_model_now and config.save_model_train_mse_threshold is not None and \
           not np.isnan(avg_train_mse) and avg_train_mse < config.save_model_train_mse_threshold:
            can_save_model_now = True
            new_train_mse_milestone_reached_this_epoch = True # Met threshold for the first time
            print(f"INFO: Train MSE {avg_train_mse:.6f} is below threshold {config.save_model_train_mse_threshold}. Model saving enabled. Validation & Reconstruction will be triggered.")
        elif config.save_model_train_mse_threshold is None: # If no threshold, saving is always possible based on metric
            can_save_model_now = True

        # Update best_train_mse_this_run (tracks the absolute best train MSE for reporting)
        if not np.isnan(avg_train_mse) and avg_train_mse < best_train_mse_this_run:
            best_train_mse_this_run = avg_train_mse

        # --- Model Saving based on Training MSE (if configured) ---
        if config.model_saving_metric == "train_mse" and can_save_model_now and not training_nan_detected:
            if not np.isnan(avg_train_mse): # Only consider valid MSE
                # Initialize best_metric_for_saving only if the threshold was JUST met this epoch
                if best_metric_for_saving == float('inf') and new_train_mse_milestone_reached_this_epoch: 
                    best_metric_for_saving = avg_train_mse
                    # Save on first qualifying metric after threshold
                    print(f"Saving model: Initial train_mse {avg_train_mse:.6f} meets criteria (threshold crossed).")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{effective_wandb_run_name}_epoch{epoch+1}_trainmse{avg_train_mse:.6f}_{timestamp}.npz"
                    model_save_path = os.path.join("../results/models", model_filename)
                    os.makedirs("../results/models", exist_ok=True)
                    pxu.save_params(model, model_save_path, filter=lambda x: isinstance(x, (pxnn.LayerParam, pxc.VodeParam)))
                    print(f"Saved model to {model_save_path}")

                elif avg_train_mse < (best_metric_for_saving - config.early_stopping_min_delta):
                    print(f"Saving model: Train MSE improved! {best_metric_for_saving:.6f} -> {avg_train_mse:.6f}")
                    best_metric_for_saving = avg_train_mse
                    new_train_mse_milestone_reached_this_epoch = True # Also a milestone for triggering val/recon
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{effective_wandb_run_name}_epoch{epoch+1}_trainmse{avg_train_mse:.6f}_{timestamp}.npz"
                    model_save_path = os.path.join("../results/models", model_filename)
                    os.makedirs("../results/models", exist_ok=True)
                    pxu.save_params(model, model_save_path, filter=lambda x: isinstance(x, (pxnn.LayerParam, pxc.VodeParam)))
                    print(f"Saved model to {model_save_path}")
            else:
                print(f"Warning: Training MSE is NaN at epoch {epoch+1}, cannot use for model saving decision.")
        
        training_nan_detected = False
        if jnp.isnan(avg_train_w_energy) or jnp.isnan(avg_train_mse):
            print(f"!!! WARNING: NaN detected in training metrics (Energy: {avg_train_w_energy}, MSE: {avg_train_mse}) at Epoch {epoch+1}. Stopping training for this run. !!!")
            early_stopped = True
            early_stop_reason = 'NaN'
            early_stopped_epoch = epoch
            training_nan_detected = True
            # Ensure metrics logged reflect the failure
            avg_train_w_energy = float('nan')
            avg_train_mse = float('nan') 
        else:
            # Update best_train_mse_this_run if current epoch's avg_train_mse is better and valid
            if not np.isnan(avg_train_mse) and avg_train_mse < best_train_mse_this_run:
                best_train_mse_this_run = avg_train_mse
        
        # Initialize epoch_metrics here to store training metrics
        epoch_metrics = {
            'Losses/train_w_energy_avg': avg_train_w_energy,
            'Losses/train_mse_avg': avg_train_mse,
            'LearningRate/weights': current_w_lr,
            'LearningRate/hidden': current_h_lr
        }

        # --- Early stopping based on Training MSE (if configured) ---
        if config.use_early_stopping and config.early_stopping_metric == "train_mse" and not training_nan_detected:
            if not np.isnan(avg_train_mse): # Only consider valid MSE
                if best_metric_for_early_stopping == float('inf'): # Initialize on first valid metric
                    best_metric_for_early_stopping = avg_train_mse
                    print(f"Initialized best_metric_for_early_stopping (train_mse): {best_metric_for_early_stopping:.6f}")
                
                if avg_train_mse < (best_metric_for_early_stopping - config.early_stopping_min_delta):
                    print(f"Training MSE improved! {best_metric_for_early_stopping:.6f} -> {avg_train_mse:.6f}")
                    best_metric_for_early_stopping = avg_train_mse
                    epochs_without_improvement_early_stopping = 0
                else:
                    epochs_without_improvement_early_stopping += 1
                    print(f"No improvement in Training MSE for early stopping for {epochs_without_improvement_early_stopping} epochs (best: {best_metric_for_early_stopping:.6f})")
                    if epochs_without_improvement_early_stopping >= config.early_stopping_patience:
                        print(f"Early stopping triggered on Training MSE after {epoch+1} epochs!")
                        early_stopped = True
                        early_stop_reason = "TrainMetric"
                        early_stopped_epoch = epoch
                        # Log metrics before breaking from the main loop
                        run.log(epoch_metrics, step=epoch+1)
                        break 
            else:
                print(f"Warning: Training MSE is NaN at epoch {epoch+1}, cannot use for early stopping decision.")


        # --- Run validation and other epoch-end tasks only if training was successful --- 
        # Trigger validation if it's a validation epoch OR if the MSE threshold was met
        trigger_validation_event = (
            ((epoch + 1) % config.validation_every_n_epochs == 0) or \
            (epoch == config.epochs - 1) or \
            (early_stopped and epoch == early_stopped_epoch) or \
            (new_train_mse_milestone_reached_this_epoch and can_save_model_now) 
        )
        
        if not training_nan_detected and trigger_validation_event and not (early_stopped and early_stop_reason == 'TrainMetric' and not ((epoch + 1) % config.validation_every_n_epochs == 0 or epoch == config.epochs -1)):
            # If early stopping was due to train metric, we might not need to run validation again,
            # unless it's a scheduled validation, the very last epoch, or a new train MSE milestone was just hit.
            # This condition tries to avoid redundant validation if only TrainMetric early stop occurred AND it's not a scheduled/final/milestone validation.

            print(f"Evaluating pretext task metrics on validation set (Epoch {epoch+1})...")

            # Determine the optimizer for hidden states during evaluation
            # It should use the current_h_lr if a schedule is active for training h,
            # otherwise, it uses the fixed config.hidden_lr_inference.
            if config.use_lr_schedule_h:
                # Create a new base optimizer for hidden states for this specific evaluation, using current_h_lr
                if config.use_adamw_for_hidden_optimizer:
                    eval_base_optim_h = optax.adamw(learning_rate=current_h_lr, b1=0.9, b2=0.999, eps=1e-8)
                else:
                    eval_base_optim_h = optax.sgd(current_h_lr, momentum=config.hidden_momentum)
                
                # Apply clipping, similar to how final_optim_h_inference was constructed
                eval_optim_h_steps = []
                if config.h_grad_clip_norm is not None and config.h_grad_clip_norm > 0:
                    # Define h_clipper if not already defined (e.g., if h_grad_clip_norm was only for training)
                    # This h_clipper is the same one used for optim_h_inference's original definition.
                    h_eval_clipper = optax.clip_by_global_norm(config.h_grad_clip_norm)
                    eval_optim_h_steps.append(h_eval_clipper)
                eval_optim_h_steps.append(eval_base_optim_h)
                final_eval_optim_h_config = optax.chain(*eval_optim_h_steps)
                
                # Wrap in pxu.Optim
                actual_optim_h_for_eval = pxu.Optim(lambda: final_eval_optim_h_config)
            else:
                # If no schedule for h, use the pre-configured optim_h_inference
                actual_optim_h_for_eval = optim_h_inference

            pretext_metrics = eval_pretext_metrics(
                val_loader, # Use validation loader
                T_values=config.eval_inference_steps, # Use eval_inference_steps
                use_corruption=config.use_corruption, 
                corrupt_ratio=config.corrupt_ratio,
                model=model, 
                optim_h=optim_h, # <<< Use the correctly configured optimizer for eval
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
            current_val_metric = pretext_metrics.get('full_mse', float('inf')) # <<< Use full_mse for early stopping
            if best_val_loss_for_overall_best_model == float('inf') and current_val_metric != float('inf'):
                 best_val_loss_for_overall_best_model = current_val_metric
                 print(f"Initialized best_val_loss_for_overall_best_model for early stopping with full_mse: {best_val_loss_for_overall_best_model:.6f}")
            
             # Early stopping check based on the chosen metric (e.g., masked_mse)
            if config.use_early_stopping and best_val_loss_for_overall_best_model != float('inf'): # Ensure best_val_loss_for_overall_best_model is initialized
                if current_val_metric < (best_val_loss_for_overall_best_model - config.early_stopping_min_delta):
                    print(f"Overall best validation MSE improved! {best_val_loss_for_overall_best_model:.6f} -> {current_val_metric:.6f}")
                    best_val_loss_for_overall_best_model = current_val_metric
                    # epochs_without_improvement_early_stopping = 0 # This is now handled by the chosen metric's block
                    # Save the best model here if desired
                    try:
                        model_save_dir = "../results/models"
                        os.makedirs(model_save_dir, exist_ok=True)
                        # New filename format
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = f"{effective_wandb_run_name}_epoch{epoch+1}_BESTVALmse{best_val_loss_for_overall_best_model:.6f}_{timestamp}.npz"
                        model_save_path = os.path.join(model_save_dir, model_filename)
                        
                        # Save LayerParams (weights) and VodeParams (hidden states like h, u)
                        pxu.save_params(model, model_save_path, filter=lambda x: isinstance(x, (pxnn.LayerParam, pxc.VodeParam)))
                        print(f"Saved new overall best validation model to {model_save_path}")
                        # Log as artifact to W&B
                        # if wandb.run and wandb.run.mode != "disabled":
                        #     model_artifact = wandb.Artifact(name=f"{effective_wandb_run_name}_best_model", type="model")
                        #     model_artifact.add_file(model_save_path)
                        #     wandb.run.log_artifact(model_artifact)
                        #     print(f"Logged best model artifact to W&B: {model_artifact.name}")
                    except Exception as e:
                        print(f"Error saving model: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # epochs_without_improvement_early_stopping += 1 # This is now handled by the chosen metric's block
                    print(f"No improvement in validation metric for {epochs_without_improvement_early_stopping} epochs (best: {best_val_loss_for_overall_best_model:.6f})")
            
            # --- Early stopping based on Validation MSE (if configured) ---
            if config.use_early_stopping and config.early_stopping_metric == "val_mse" and not training_nan_detected: # ensure not already stopped by NaN
                if 'full_mse' in pretext_metrics and not np.isnan(pretext_metrics['full_mse']):
                    current_val_mse_for_stopping = pretext_metrics['full_mse']
                    
                    if best_metric_for_early_stopping == float('inf'): # Initialize on first valid metric
                        best_metric_for_early_stopping = current_val_mse_for_stopping
                        print(f"Initialized best_metric_for_early_stopping (val_mse): {best_metric_for_early_stopping:.6f}")

                    if current_val_mse_for_stopping < (best_metric_for_early_stopping - config.early_stopping_min_delta):
                        print(f"Validation MSE for early stopping improved! {best_metric_for_early_stopping:.6f} -> {current_val_mse_for_stopping:.6f}")
                        best_metric_for_early_stopping = current_val_mse_for_stopping
                        epochs_without_improvement_early_stopping = 0
                    else:
                        epochs_without_improvement_early_stopping += 1
                        print(f"No improvement in Validation MSE for early stopping for {epochs_without_improvement_early_stopping} epochs (best: {best_metric_for_early_stopping:.6f})")
                        if epochs_without_improvement_early_stopping >= config.early_stopping_patience:
                            print(f"Early stopping triggered on Validation MSE after {epoch+1} epochs!")
                            early_stopped = True
                            early_stop_reason = "ValidationMetric" # More specific reason
                            early_stopped_epoch = epoch
                            # Log metrics before breaking from the main loop (if not already logged)
                            run.log(epoch_metrics, step=epoch+1)
                            break # Break from the main epoch loop
                else:
                    print(f"Warning: Validation MSE ('full_mse') is NaN or not available at epoch {epoch+1}, cannot use for early stopping decision.")

        
        # Log metrics collected so far for this epoch (includes train loss, potentially val metrics)
        run.log(epoch_metrics, step=epoch+1)  # Use epoch+1 to start from step 1

        # --- Linear Probing ---
        if config.linear_probe_every_n_epochs > 0 and (epoch + 1) % config.linear_probe_every_n_epochs == 0:
            print(f"--- Running Linear Probe at Epoch {epoch+1} ---")
            # Create dataloaders specifically for probing (no SSL augmentations)
            train_loader_probe, test_loader_probe = get_debug_dataloaders(
                dataset_name=config.dataset,
                batch_size=config.batch_size, # Use MAIN training batch size for feature extraction
                root_path=config.data_dir,
                train_subset_n=config.train_subset, # Can be full dataset or a subset
                test_subset_n=config.test_subset,   # Can be full dataset or a subset
                target_class=config.target_class,
                use_ssl_augmentations=False, # IMPORTANT: Turn off augmentations for probing
                use_cifar10_norm=config.use_cifar10_norm # Keep original norm
            )
            print(f"Probe Dataloaders: Train size {len(train_loader_probe.dataset)}, Test size {len(test_loader_probe.dataset)}")

            probe_config_overrides_dict = {
                "linear_probe_vode_indices": config.linear_probe_vode_indices,
                "linear_probe_concatenate_features": config.linear_probe_concatenate_features,
                "linear_probe_use_gap": config.linear_probe_use_gap,
                "linear_probe_lr": config.linear_probe_lr,
                "linear_probe_wd": config.linear_probe_wd,
                "linear_probe_epochs": config.linear_probe_epochs,
                "linear_probe_batch_size": config.linear_probe_batch_size,
                "linear_probe_h_lr": config.linear_probe_h_lr if config.linear_probe_h_lr is not None else config.hidden_lr_inference,
                "linear_probe_inference_steps": config.linear_probe_inference_steps if config.linear_probe_inference_steps is not None else config.inference_steps,
                "linear_probe_seed": config.linear_probe_seed
            }
            
            # Ensure the model passed to probing is the current state of the model
            # The 'model' variable in this scope is the one being trained.
            # The 'model_config' is the TransformerConfig for architecture.
            current_probe_accuracy = run_linear_probe_evaluation(
                model=model,
                model_config_obj=config, # Pass the main ModelConfig
                transformer_arch_config=model_config, # Pass the TransformerConfig
                probe_config_overrides=probe_config_overrides_dict,
                train_loader=train_loader_probe,
                test_loader=test_loader_probe,
                current_epoch=epoch + 1
            )
            print(f"Linear Probe Accuracy at Epoch {epoch+1}: {current_probe_accuracy:.4f}")
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({f"Metrics/LinearProbe_Accuracy": current_probe_accuracy}, step=epoch+1)
            
            if current_probe_accuracy > best_linear_probe_accuracy_overall:
                best_linear_probe_accuracy_overall = current_probe_accuracy
                print(f"New best overall linear probe accuracy: {best_linear_probe_accuracy_overall:.4f}")
        # --- End Linear Probing ---

        # Generate reconstructions every N epochs (and for the final epoch)
        # Base the condition on one of the new validation metrics if available and valid
        # Use the most recently calculated validation metric if available
        current_val_metric_for_recon = pretext_metrics.get('full_mse', float('inf')) 

        # Trigger reconstruction based on interval OR significant improvement OR final epoch/stop OR MSE threshold
        trigger_reconstruction_event = False
        if ((epoch + 1) % config.reconstruction_every_n_epochs == 0):
            trigger_reconstruction_event = True
        # elif best_val_loss_for_overall_best_model != float('inf') and current_val_metric_for_recon < best_val_loss_for_overall_best_model - config.early_stopping_min_delta: 
        #     trigger_reconstruction_event = True # This was for triggering on any val improvement, simplifying now
        elif epoch == config.epochs - 1: # Final epoch
            trigger_reconstruction_event = True
        elif early_stopped and epoch == early_stopped_epoch: # After early stopping
             trigger_reconstruction_event = True
        elif (new_train_mse_milestone_reached_this_epoch and can_save_model_now): # New condition
            trigger_reconstruction_event = True
        # mse_threshold_met is no longer used here to trigger reconstruction
             
        if trigger_reconstruction_event and not training_nan_detected:
            print(f"Generating reconstructions for epoch {epoch+1} (Current Val Metric for Recon: {current_val_metric_for_recon:.6f}, Train MSE: {avg_train_mse:.6f})...")
            
            # Determine the optimizer for hidden states for reconstruction visualization
            # Similar logic as for pretext_metrics evaluation
            if config.use_lr_schedule_h:
                if config.use_adamw_for_hidden_optimizer:
                    recon_base_optim_h = optax.adamw(learning_rate=current_h_lr, b1=0.9, b2=0.999, eps=1e-8)
                else:
                    recon_base_optim_h = optax.sgd(current_h_lr, momentum=config.hidden_momentum)
                
                recon_optim_h_steps = []
                if config.h_grad_clip_norm is not None and config.h_grad_clip_norm > 0:
                    h_recon_clipper = optax.clip_by_global_norm(config.h_grad_clip_norm)
                    recon_optim_h_steps.append(h_recon_clipper)
                recon_optim_h_steps.append(recon_base_optim_h)
                final_recon_optim_h_config = optax.chain(*recon_optim_h_steps)
                actual_optim_h_for_recon = pxu.Optim(lambda: final_recon_optim_h_config)
            else:
                actual_optim_h_for_recon = optim_h_inference

            vis_path, vis_logs = visualize_reconstruction(
                model, 
                model_config, 
                optim_h=optim_h, # <<< Use the correctly configured optimizer for recon
                dataloader=val_loader, 
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
    
    # --- Final Model Save Check (based on absolute improvement over best_metric_for_saving) ---
    if can_save_model_now and not training_nan_detected: # Ensure saving is enabled and run wasn't a NaN failure
        final_metric_value_at_run_end = float('inf')
        metric_type_for_final_save = ""
        # epoch is 0-indexed from the loop, so epoch + 1 is the last completed epoch number (1-indexed)
        last_epoch_num_completed = epoch + 1 

        current_final_train_mse = float('inf')
        if 'epoch_metrics' in locals() and 'Losses/train_mse_avg' in epoch_metrics:
            candidate_train_mse = epoch_metrics['Losses/train_mse_avg']
            if not np.isnan(candidate_train_mse):
                current_final_train_mse = candidate_train_mse

        current_final_val_mse = float('inf')
        if 'pretext_metrics' in locals() and 'full_mse' in pretext_metrics:
            candidate_val_mse = pretext_metrics.get('full_mse', float('inf'))
            if not np.isnan(candidate_val_mse):
                current_final_val_mse = candidate_val_mse

        if config.model_saving_metric == "train_mse":
            final_metric_value_at_run_end = current_final_train_mse
            metric_type_for_final_save = "trainmse"
        elif config.model_saving_metric == "val_mse":
            final_metric_value_at_run_end = current_final_val_mse
            metric_type_for_final_save = "valmse"
        
        if metric_type_for_final_save and not np.isnan(final_metric_value_at_run_end) and final_metric_value_at_run_end < best_metric_for_saving:
            print(f"INFO: End-of-run check. Final {metric_type_for_final_save} ({final_metric_value_at_run_end:.6f}) "
                  f"is absolutely better than the best saved during run ({best_metric_for_saving:.6f}). "
                  f"Saving final model.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{effective_wandb_run_name}_epoch{last_epoch_num_completed}_finalabs{metric_type_for_final_save}{final_metric_value_at_run_end:.6f}_{timestamp}.npz"
            model_save_path = os.path.join("../results/models", model_filename)
            os.makedirs("../results/models", exist_ok=True)
            try:
                pxu.save_params(model, model_save_path, filter=lambda x: isinstance(x, (pxnn.LayerParam, pxc.VodeParam)))
                print(f"Saved final model (absolute improvement) to {model_save_path}")
            except Exception as e:
                print(f"Error saving final model (absolute improvement): {e}")
    
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
    if wandb_run is None:  # Only finish if we created the run ourselves
        wandb.finish()
    
    print("Debug run completed!")
    # Return the average training MSE of the last epoch, or infinity if NaN occurred
    final_avg_train_mse = float('inf') # Default to infinity
    if 'epoch_metrics' in locals() and 'Losses/train_mse_avg' in epoch_metrics:
        # epoch_metrics now guaranteed to contain Python float or NaN
        final_avg_train_mse = epoch_metrics['Losses/train_mse_avg'] 
    elif 'avg_train_mse' in locals() and not np.isnan(avg_train_mse):
        # Fallback if epoch_metrics wasn't populated but avg_train_mse from the last loop iteration is valid
        final_avg_train_mse = float(avg_train_mse) # Ensure fallback is float
    
    # If a NaN was detected during training, explicitly set final_avg_train_mse to NaN for clarity
    if early_stop_reason == 'NaN':
        final_avg_train_mse = float('nan') # Use NaN to be distinct from other high MSEs
    
    print(f"Run {effective_wandb_run_name} finished with final avg_train_mse: {final_avg_train_mse}")
    
    print(f"DEBUG: Returning from run_experiment - final_mse: {final_avg_train_mse}, Type: {type(final_avg_train_mse)}, Reason: {early_stop_reason}")
    # Ensure best_val_loss_for_overall_best_model is defined even if validation never runs or fails
    if 'best_val_loss_for_overall_best_model' not in locals():
        best_val_loss_for_overall_best_model = float('inf')
    # Ensure best_train_mse_this_run is defined even if training loop doesn't run (e.g. epochs=0)
    if 'best_train_mse_this_run' not in locals():
        best_train_mse_this_run = float('inf')
        
    # Ensure best_val_loss_for_overall_best_model is a Python float before returning
    if hasattr(best_val_loss_for_overall_best_model, 'item'): # Check if it's a JAX scalar or 0-dim array
        best_val_loss_for_overall_best_model = float(best_val_loss_for_overall_best_model.item())
    else:
        best_val_loss_for_overall_best_model = float(best_val_loss_for_overall_best_model) # Ensure it's a float otherwise

    return best_val_loss_for_overall_best_model, best_train_mse_this_run, final_avg_train_mse, early_stop_reason, best_linear_probe_accuracy_overall

def run_sweep():
    """Main function for running with wandb sweeps."""
    # Initialize wandb first to get access to wandb.config
    run = wandb.init()
    
    # Get base config name from sweep config
    base_config_name = wandb.config.get('config', DEFAULT_CONFIG)
    
    # Create config overrides from wandb.config
    # Skip 'config' key as it's used for base config selection
    config_overrides = {k: v for k, v in wandb.config.items() if k != 'config'}
    
    print(f"Running sweep with base config: {base_config_name}")
    print(f"Sweep overrides: {config_overrides}")
    
    # Run the experiment with sweep parameters
    best_val_mse, best_train_mse, final_train_mse, early_stop_reason, best_probe_acc = run_experiment(
        base_config_name=base_config_name,
        config_overrides=config_overrides,
        wandb_project_name=None,  # Will use the sweep's project
        wandb_run_name=None,      # Will use the sweep's run name
        wandb_mode="online",      # Sweeps should always be online
        wandb_run=run             # Pass the existing run
    )
    
    # Log the final metrics that the sweep will optimize using the run object directly
    run.log({
        "best_val_mse": best_val_mse,
        "best_train_mse": best_train_mse, 
        "final_train_mse": final_train_mse,
        "best_probe_accuracy": best_probe_acc,
        "early_stop_reason": early_stop_reason
    })
    
    print(f"Sweep run completed. Best val MSE: {best_val_mse:.6f}")
    return best_val_mse

if __name__ == "__main__":
    cli_args = parse_args()
    
    # Prepare config_overrides from CLI arguments, excluding 'config' which is used for base_config_name
    overrides = {k: v for k, v in vars(cli_args).items() if v is not None and k != 'config'}
    
    if cli_args.sweep:
        run_sweep()
    else:
        run_experiment(base_config_name=cli_args.config, 
                       config_overrides=overrides) 