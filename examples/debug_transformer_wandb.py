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
import contextlib
import re

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import the TransformerDecoder and utility functions
from src.decoder_transformer import (
    TransformerDecoder, 
    TransformerConfig,
    train, 
    eval, 
    eval_on_batch_partial,
    visualize_reconstruction,
    model_debugger,
    forward,
    train_on_batch
)

@dataclass
class ModelConfig:
    """Configuration for transformer models with all hyperparameters in one place."""
    name: str
    # Dataset settings
    dataset: str = "cifar10"
    data_dir: str = "../datasets/"
    train_subset: int = 5
    test_subset: int = 5
    target_class: Optional[int] = None
    
    # Model architecture
    latent_dim: int = 128
    hidden_size: int = 48
    num_heads: int = 6
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    patch_size: int = 4
    axes_dim: List[int] = field(default_factory=lambda: [16, 16])
    theta: int = 10_000
    use_noise: bool = True
    
    # Training settings
    batch_size: int = 1
    epochs: int = 2
    inference_steps: int = 8
    peak_lr_weights: float = 1e-3
    peak_lr_hidden: float = 0.01
    weight_decay: float = 2e-4
    warmup_epochs: int = 1
    use_lr_schedule: bool = True  # New option to control whether to use LR scheduling
    seed: int = 42
    
    # Visualization settings
    num_images: int = 5

# Predefined configurations for easy experimentation
MODEL_CONFIGS = {
    "debug_tiny": ModelConfig(
        name="debug_tiny",
        batch_size=1,
        latent_dim=64,
        hidden_size=128,
        num_heads=4,
        num_blocks=6,
        train_subset=5,
        test_subset=5,
        inference_steps=8,
        use_lr_schedule=False
    ),
    "debug_small": ModelConfig(
        name="debug_small",
        batch_size=1,
        latent_dim=128,
        hidden_size=128,
        num_heads=6, 
        num_blocks=2,
        train_subset=10,
        test_subset=10,
        inference_steps=8,
        use_lr_schedule=True
    ),
    "baseline": ModelConfig(
        name="baseline",
        batch_size=10,
        latent_dim=256,
        hidden_size=128,
        num_heads=8,
        num_blocks=6,
        train_subset=100,
        test_subset=100,
        inference_steps=16,
        peak_lr_weights=5e-4,
        peak_lr_hidden=0.005,
        use_lr_schedule=True
    )
}

# Default configuration to use
DEFAULT_CONFIG = "debug_small"

def create_config(dataset="cifar10", latent_dim=128, num_blocks=1, hidden_size=48, num_heads=6,
                 mlp_ratio=4.0, patch_size=4, axes_dim=None, theta=10_000, use_noise=True):
    """Create a TransformerConfig based on the dataset name and parameters."""
    axes_dim = axes_dim or [16, 16]
    
    if dataset == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
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
            use_noise=use_noise
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def parse_args():
    parser = argparse.ArgumentParser(description='Debug a transformer model with W&B logging')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help=f'Predefined configuration to use. Options: {", ".join(MODEL_CONFIGS.keys())}')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Latent dimension size')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Hidden dimension size')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--num-blocks', type=int, default=None,
                        help='Number of transformer blocks')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--inference-steps', type=int, default=None,
                        help='Number of inference steps')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory to store datasets')
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Number of samples to use from the training set')
    parser.add_argument('--test-subset', type=int, default=None,
                        help='Number of samples to use from the test set')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Filter the dataset to a specific class (0-9 for CIFAR-10)')
    parser.add_argument('--peak-lr-weights', type=float, default=None,
                        help='Peak learning rate for weights')
    parser.add_argument('--peak-lr-hidden', type=float, default=None,
                        help='Peak learning rate for hidden states')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Number of warmup epochs for learning rate')
    parser.add_argument('--use-lr-schedule', action='store_true', dest='use_lr_schedule',
                        help='Use learning rate schedule with warmup and decay')
    parser.add_argument('--no-lr-schedule', action='store_false', dest='use_lr_schedule',
                        help='Use constant learning rate without scheduling')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-images', type=int, default=None,
                      help='Number of images to use for reconstruction visualization')
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
        num_workers=2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
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

def analyze_pcx_gradients(h_grad, w_grad):
    """Analyze PCX gradients with proper PCX-specific extraction."""
    
    # Get all leaf nodes with gradients from PCX gradient structure
    def extract_pcx_gradients(grad_dict, prefix=""):
        """Extract all gradient arrays from PCX gradient structure."""
        if grad_dict is None:
            return {}
            
        # Top level in PCX grad structure is usually {"model": {...}}
        if isinstance(grad_dict, dict) and "model" in grad_dict:
            return extract_pcx_gradients(grad_dict["model"], prefix)
            
        gradients = {}
        
        if not isinstance(grad_dict, dict):
            # For JAX arrays, extract directly
            if hasattr(grad_dict, 'shape'):
                return {prefix: grad_dict}
            return {}
            
        # Process nested dictionaries
        for key, value in grad_dict.items():
            component_name = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                sub_gradients = extract_pcx_gradients(value, component_name)
                gradients.update(sub_gradients)
            elif hasattr(value, 'shape'):
                # Direct gradient array
                gradients[component_name] = value
                
        return gradients
    
    # Extract gradients from PCX structure
    h_grads = extract_pcx_gradients(h_grad, "hidden")
    w_grads = extract_pcx_gradients(w_grad, "weight")
    
    print(f"Found {len(h_grads)} hidden gradient arrays and {len(w_grads)} weight gradient arrays")
    
    # Calculate statistics for each gradient array
    def calculate_grad_stats(grad_dict):
        stats = {}
        for name, grad_array in grad_dict.items():
            flat_grad = jnp.ravel(grad_array)
            if flat_grad.size == 0:
                continue
                
            stats[name] = {
                'norm': float(jnp.linalg.norm(flat_grad)),
                'mean': float(jnp.mean(flat_grad)),
                'std': float(jnp.std(flat_grad)),
                'max': float(jnp.max(flat_grad)),
                'min': float(jnp.min(flat_grad)),
                'shape': grad_array.shape,
                'size': flat_grad.size
            }
        return stats
    
    h_stats = calculate_grad_stats(h_grads)
    w_stats = calculate_grad_stats(w_grads)
    
    # Calculate total norm across all arrays
    def calculate_total_norm(stats_dict):
        squared_sum = sum(stats['norm']**2 for stats in stats_dict.values())
        return jnp.sqrt(squared_sum) if squared_sum > 0 else 0.0
        
    h_total_norm = calculate_total_norm(h_stats)
    w_total_norm = calculate_total_norm(w_stats)
    
    print(f"Hidden gradient norm: {h_total_norm}")
    print(f"Weight gradient norm: {w_total_norm}")
    
    # Find maximum component norm
    h_max_component = max([s['norm'] for s in h_stats.values()], default=0.0)
    w_max_component = max([s['norm'] for s in w_stats.values()], default=0.0)
    
    # Find top components by norm
    h_top_components = sorted(
        [(k, s['norm']) for k, s in h_stats.items()],
        key=lambda x: x[1], reverse=True
    )[:5]
    
    w_top_components = sorted(
        [(k, s['norm']) for k, s in w_stats.items()],
        key=lambda x: x[1], reverse=True
    )[:5]
    
    print("Top hidden gradient components:")
    for name, norm in h_top_components:
        print(f"  {name}: {norm}")
        
    print("Top weight gradient components:")
    for name, norm in w_top_components:
        print(f"  {name}: {norm}")
    
    # Calculate ratio if possible
    h_to_w_ratio = h_total_norm / w_total_norm if w_total_norm > 0 else 0.0
    
    return {
        'components': {
            'hidden': h_stats,
            'weight': w_stats
        },
        'summary': {
            'hidden': {
                'total_norm': h_total_norm,
                'max_component_norm': h_max_component,
                'component_count': len(h_stats)
            },
            'weight': {
                'total_norm': w_total_norm,
                'max_component_norm': w_max_component,
                'component_count': len(w_stats)
            },
            'h_to_w_ratio': h_to_w_ratio
        },
        'top_components': {
            'hidden': h_top_components,
            'weight': w_top_components
        }
    }

def visualize_reconstruction(model, optim_h, dataloader, T_values=[24], use_corruption=False, corrupt_ratio=0.5, target_class=None, num_images=2, wandb_run=None, epoch=None, step=None):
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Import or define STATUS_FORWARD constant
    STATUS_FORWARD = "forward"
    
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
                    if hasattr(model, 'final_layer'):
                        # The TransformerDecoder doesn't have a 'final_layer' attribute directly
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
    
    # Create epoch string for filename
    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../results", exist_ok=True)
    reconstruction_path = f"../results/reconstruction{epoch_str}_{timestamp}.png"
    plt.savefig(reconstruction_path)
    
    # Create a dictionary for the caller to log instead of logging directly
    log_dict = {}
    
    # Only prepare image for W&B if a run is provided
    if wandb_run is not None:
        # Create log key with epoch info
        log_key = f"reconstructions{epoch_str}" if epoch is not None else "reconstructions"
        log_dict[log_key] = wandb.Image(reconstruction_path)
        
    plt.close(fig)
    return orig_images, recon_images, reconstruction_path, log_dict

def log_vode_stats(model, h_grad, w_grad, run, epoch):
    """
    Log detailed statistics about each Vode's energy and gradients to wandb.
    
    Args:
        model: The TransformerDecoder model
        h_grad: Hidden state gradients
        w_grad: Weight gradients
        run: Wandb run object
        epoch: Current epoch
    """
    # Dictionary to collect all stats for wandb
    vode_stats = {}
    
    # Extract individual energy contributions from each Vode
    vode_energies = model.get_submodule_energies()
    print("vode_energies: ", vode_energies)

    # for i, vode in enumerate(model.vodes):
    #     try:
    #         print("vode.cache['E']: ", vode.cache)
    #         vode_energies.append(vode.cache)

    #     except Exception as e:
    #         print(f"Error calculating energy for vode {i}: {e}")
    #         vode_energies.append(None)

    # print("h_grad: ", h_grad)
    
    # Extract gradients per Vode
    vode_grad_norms = extract_vode_gradient_norms(h_grad)
    
    # Create wandb logging dictionaries
    for i, (energy, grad_norm) in enumerate(zip(vode_energies, vode_grad_norms)):
        if energy is not None:
            vode_stats[f"VodeEnergy/vode_{i}"] = energy
        
        if grad_norm is not None:
            vode_stats[f"VodeGradNorm/vode_{i}"] = grad_norm
    
    # Create a bar chart for energy distribution
    energy_data = [[i, e] for i, e in enumerate(vode_energies) if e is not None]
    if energy_data:
        energy_table = wandb.Table(data=energy_data, columns=["vode_index", "energy"])
        vode_stats["VodeEnergy/distribution"] = wandb.plot.bar(
            energy_table, "vode_index", "energy", title="Energy by Vode"
        )
    
    # Create a bar chart for gradient norm distribution
    grad_data = [[i, n] for i, n in enumerate(vode_grad_norms) if n is not None]
    if grad_data:
        grad_table = wandb.Table(data=grad_data, columns=["vode_index", "grad_norm"])
        vode_stats["VodeGradNorm/distribution"] = wandb.plot.bar(
            grad_table, "vode_index", "grad_norm", title="Gradient Norm by Vode"
        )
    
    # Log everything to wandb
    run.log(vode_stats, step=epoch+1)
    
    return vode_stats

def extract_vode_gradient_norms(h_grad):
    """Simplified direct extraction based on the known structure"""
    vode_grad_norms = []
    
    try:
        if isinstance(h_grad, dict) and "model" in h_grad:
            model = h_grad["model"]
            
            # Get the vodes attribute
            for i in range(len(model.vodes)):
                try:
                    # Try to access vodes[i].h directly
                    vode_param_name = f"vodes[{i}].h"
                    if hasattr(model, vode_param_name):
                        vode_param = getattr(model, vode_param_name)
                        if hasattr(vode_param, 'value'):
                            grad_array = vode_param.value
                        else:
                            grad_array = vode_param
                            
                        if hasattr(grad_array, 'flatten'):
                            flat_grad = grad_array.flatten()
                            norm = float(jnp.linalg.norm(flat_grad))
                            vode_grad_norms.append(norm)
                        else:
                            vode_grad_norms.append(None)
                    else:
                        # If we don't find this vode, we've reached the end
                        break
                except Exception as e:
                    print(f"Error accessing vode {i}: {e}")
                    vode_grad_norms.append(None)
    except Exception as e:
        print(f"Error in direct extraction: {e}")
    
    return vode_grad_norms


def main():
    """Main function to run the debugging process with W&B logging."""
    args = parse_args()
    
    # Load the base configuration
    config = MODEL_CONFIGS[args.config]
    
    # Override configuration with any command-line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config':
            setattr(config, arg_name, arg_value)
    
    # Print the effective configuration
    print(f"\nUsing configuration '{config.name}' with settings:")
    for key, value in vars(config).items():
        if key != 'name':
            print(f"  {key}: {value}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    
    # Initialize Weights & Biases
    run = wandb.init(
        entity="neural-machines",
        project="debug-transformer",
        config=vars(config),
        mode="online"  # Change to online for immediate verification
    )
    
    # Upload code artifacts to W&B
    try:
        code_artifact = wandb.Artifact(name="source_code", type="code")
        
        # Add the current file
        code_artifact.add_file(__file__)
        
        # Add the decoder_transformer.py file
        decoder_transformer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/decoder_transformer.py")
        if os.path.exists(decoder_transformer_path):
            code_artifact.add_file(decoder_transformer_path)
            print(f"Added decoder_transformer.py to W&B artifact")
        else:
            print(f"Warning: Could not find decoder_transformer.py at {decoder_transformer_path}")
        
        # Log the artifact to W&B
        run.log_artifact(code_artifact)
        print("Uploaded code artifacts to W&B")
    except Exception as e:
        print(f"Error uploading code to W&B: {e}")
    
    print(f"Creating configuration for debugging CIFAR-10 transformer...")
    model_config = create_config(
        dataset=config.dataset,
        latent_dim=config.latent_dim,
        num_blocks=config.num_blocks,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        patch_size=config.patch_size,
        axes_dim=config.axes_dim,
        theta=config.theta,
        use_noise=config.use_noise
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
    optim_h = pxu.Optim(lambda: optax.sgd(hidden_lr_fn, momentum=0.9))
    optim_w = pxu.Optim(
        lambda: optax.adamw(weights_lr_fn, weight_decay=config.weight_decay), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    # Store validation losses
    val_losses = []
    
    # # Direct inspection of a single training batch to understand gradient structure
    # print("\n=== Direct Inspection of train_on_batch Output ===")
    # test_batch = next(iter(train_loader.dataloader))
    # x_test, _ = test_batch
    # x_test = jnp.array(x_test.numpy())
    
    # print("Calling train_on_batch directly to inspect outputs...")
    # h_energy, w_energy, h_grad, w_grad = train_on_batch(
    #     1, x_test, model=model, optim_w=optim_w, optim_h=optim_h, epoch=0, step=0
    # )
    
    # print(f"h_energy: {h_energy}, w_energy: {w_energy}")
    # print(f"Are energies identical? {h_energy == w_energy}")
    
    # print("\nInspecting hidden gradient structure:")
    # h_grad_type = type(h_grad)
    # print(f"h_grad type: {h_grad_type}")
    
    # if isinstance(h_grad, dict):
    #     print(f"h_grad keys: {h_grad.keys()}")
    #     for key in h_grad.keys():
    #         print(f"  h_grad[{key}] type: {type(h_grad[key])}")
    #         if isinstance(h_grad[key], dict):
    #             print(f"    h_grad[{key}] keys: {h_grad[key].keys()}")
    
    # print("\nInspecting weight gradient structure:")
    # w_grad_type = type(w_grad)
    # print(f"w_grad type: {w_grad_type}")
    
    # if isinstance(w_grad, dict):
    #     print(f"w_grad keys: {w_grad.keys()}")
    #     for key in w_grad.keys():
    #         print(f"  w_grad[{key}] type: {type(w_grad[key])}")
    #         if isinstance(w_grad[key], dict):
    #             print(f"    w_grad[{key}] keys: {w_grad[key].keys()}")
    
    # print("=== End of Direct Inspection ===\n")
    
    print(f"Training for {config.epochs} epochs with W&B logging...")
    
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
        
        # Train for one epoch
        h_energy, w_energy, h_grad, w_grad = train(train_loader, config.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch)
        
        # Log detailed vode statistics
        vode_stats = log_vode_stats(model, h_grad, w_grad, run, epoch)
        
        # Evaluate on validation set
        val_loss = eval(train_loader, config.inference_steps, model=model, optim_h=optim_h)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.6f}")
        
        # Collect all metrics for this epoch in a single dictionary
        epoch_metrics = {
            'Losses/validation_loss': val_loss,
            'LearningRate/weights': current_w_lr,
            'LearningRate/hidden': current_h_lr
        }
        
        # # Generate reconstructions every 5 epochs (and for the final epoch)
        # if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
        #     print(f"Generating reconstructions for epoch {epoch+1}...")
        #     _, _, recon_path, recon_logs = visualize_reconstruction(
        #         model, 
        #         optim_h, 
        #         val_loader, 
        #         T_values=[1, 2, 4, 8, 16, 32], 
        #         use_corruption=False,
        #         num_images=config.num_images,
        #         wandb_run=run,
        #         epoch=epoch+1
        #     )
        #     # Add reconstruction logs to the epoch metrics
        #     epoch_metrics.update(recon_logs)
        
        # Add system metrics
        epoch_metrics.update({
            'System/GPU_Memory_Allocated': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'System/GPU_Memory_Cached': torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
            'System/CPU_Memory_Usage': psutil.Process().memory_info().rss / 1024**2
        })
        
        # Log all metrics for this epoch with a consistent step number
        run.log(epoch_metrics, step=epoch+1)  # Use epoch+1 to start from step 1
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")
    
    # Close W&B after all logging is complete
    wandb.finish()
    
    # Plot validation loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.epochs + 1), val_losses, marker='o')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    os.makedirs("../debug_logs", exist_ok=True)
    plt.savefig("../debug_logs/validation_loss_curve.png")
    
    print("Debug run completed!")

if __name__ == "__main__":
    main() 