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

def create_config(dataset="cifar10", latent_dim=128, num_blocks=1):
    """Create a TransformerConfig based on the dataset name."""
    if dataset == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=48,
            num_heads=6,
            num_blocks=num_blocks,
            mlp_ratio=4.0,
            patch_size=4,
            axes_dim=[16, 16],
            theta=10_000,
            use_noise=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def parse_args():
    parser = argparse.ArgumentParser(description='Debug a transformer model with W&B logging')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension size (default: 128)')
    parser.add_argument('--num-blocks', type=int, default=6,
                        help='Number of transformer blocks (default: 1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--inference-steps', type=int, default=8,
                        help='Number of inference steps (default: 4)')
    parser.add_argument('--data-dir', type=str, default='../datasets/',
                        help='Directory to store datasets (default: ../datasets/)')
    parser.add_argument('--train-subset', type=int, default=5,
                        help='Number of samples to use from the training set (default: 1)')
    parser.add_argument('--test-subset', type=int, default=5,
                        help='Number of samples to use from the test set (default: 1)')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Filter the dataset to a specific class (0-9 for CIFAR-10)')
    parser.add_argument('--peak-lr-weights', type=float, default=1e-3,
                        help='Peak learning rate for weights (default: 1e-3)')
    parser.add_argument('--peak-lr-hidden', type=float, default=0.01,
                        help='Peak learning rate for hidden states (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=2e-4,
                        help='Weight decay for AdamW optimizer (default: 2e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs for learning rate (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-images', type=int, default=5,
                      help='Number of images to use for reconstruction visualization (default: 5)')
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

def track_batch_dynamics(model, optim_h, optim_w, batch, inference_steps, epoch):
    """Track dynamics of a single batch with detailed logging."""
    x, y = batch
    x_numpy = x.numpy()
    x = jnp.array(x_numpy)
    
    # Track gradients and errors over inference steps
    gradient_history = []
    error_history = []
    component_history = []
    
    # Flag to track if component plot was created
    component_plot_created = False
    
    # Process each inference step
    for step in range(inference_steps):
        # Training step
        h_energy, w_energy, h_grad, w_grad = train_on_batch(
            1, x, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=step
        )
        
        # Print energy values for debugging
        print(f"Step {step} - h_energy: {h_energy}, w_energy: {w_energy}")
        print(f"Step {step} - h_energy type: {type(h_energy)}, w_energy type: {type(w_energy)}")
        
        # Convert energies to float for consistent handling
        if hasattr(h_energy, 'item'):
            h_energy = float(h_energy.item())
        else:
            h_energy = float(h_energy)
            
        if hasattr(w_energy, 'item'):
            w_energy = float(w_energy.item()) 
        else:
            w_energy = float(w_energy)
        
        # Analyze PCX gradients
        grad_stats = analyze_pcx_gradients(h_grad, w_grad)
        
        # Store component norms for plotting
        step_components = {'step': step}
        
        # Extract top components by gradient norm
        h_components = sorted(
            [(k, v['norm']) for k, v in grad_stats['components']['hidden'].items()],
            key=lambda x: x[1], reverse=True
        )[:5]  # Top 5 hidden components
        
        w_components = sorted(
            [(k, v['norm']) for k, v in grad_stats['components']['weight'].items()],
            key=lambda x: x[1], reverse=True
        )[:5]  # Top 5 weight components
        
        # Add to tracking dictionary
        for k, v in h_components:
            step_components[f"h_{k.split('.')[-1]}"] = v
            
        for k, v in w_components:
            step_components[f"w_{k.split('.')[-1]}"] = v
            
        component_history.append(step_components)
        
        # Track gradients and errors
        step_metrics = {
            'step': step,
            'h_energy': h_energy,
            'w_energy': w_energy,
            'h_grad_norm': grad_stats['summary']['hidden']['total_norm'],
            'w_grad_norm': grad_stats['summary']['weight']['total_norm'],
            'h_to_w_ratio': grad_stats['summary']['h_to_w_ratio'],
            'h_max_component': grad_stats['summary']['hidden']['max_component_norm'],
            'w_max_component': grad_stats['summary']['weight']['max_component_norm']
        }
        gradient_history.append(step_metrics)
        
        # Calculate reconstruction error
        loss, x_hat = eval_on_batch_partial(
            use_corruption=False, corrupt_ratio=0.5, T=1, 
            x=x, model=model, optim_h=optim_h
        )
        error_history.append({
            'step': step,
            'loss': float(loss),
            'reconstruction_error': float(jnp.mean(jnp.square(x - x_hat[1])))
        })
    
    # Create visualizations
    debug_dir = "../debug_logs/dynamics"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Define file paths for plots
    gradient_dynamics_path = f"{debug_dir}/gradient_dynamics_epoch{epoch}.png"
    component_gradients_path = f"{debug_dir}/component_gradients_epoch{epoch}.png"
    error_dynamics_path = f"{debug_dir}/error_dynamics_epoch{epoch}.png"
    gradient_ratios_path = f"{debug_dir}/gradient_ratios_epoch{epoch}.png"
    
    # Plot energy dynamics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    steps = [m['step'] for m in gradient_history]
    h_energy = [m['h_energy'] for m in gradient_history]
    w_energy = [m['w_energy'] for m in gradient_history]
    plt.plot(steps, h_energy, label='Hidden Energy')
    plt.plot(steps, w_energy, label='Weight Energy')
    plt.title('Energy Dynamics Over Steps')
    plt.xlabel('Inference Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    
    # Plot gradient norms
    plt.subplot(1, 2, 2)
    h_grad_norm = [m['h_grad_norm'] for m in gradient_history]
    w_grad_norm = [m['w_grad_norm'] for m in gradient_history]
    h_to_w_ratio = [m['h_to_w_ratio'] for m in gradient_history]
    
    # Check if we have valid values for log scale
    if any(v > 0 for v in h_grad_norm) or any(v > 0 for v in w_grad_norm):
        plt.semilogy(steps, h_grad_norm, label='Hidden Grad Norm')
        plt.semilogy(steps, w_grad_norm, label='Weight Grad Norm')
    else:
        plt.plot(steps, h_grad_norm, label='Hidden Grad Norm')
        plt.plot(steps, w_grad_norm, label='Weight Grad Norm')
    
    plt.title('Gradient Norm Dynamics')
    plt.xlabel('Inference Step')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(gradient_dynamics_path)
    plt.close()
    
    # Plot component gradients
    if len(component_history) > 0 and len(component_history[0]) > 1:
        # Extract component keys (excluding 'step')
        component_keys = [k for k in component_history[0].keys() if k != 'step']
        if component_keys:
            plt.figure(figsize=(15, 8))
            
            # Hidden components
            h_keys = [k for k in component_keys if k.startswith('h_')]
            has_positive_h_values = False
            
            if h_keys:
                plt.subplot(1, 2, 1)
                for key in h_keys:
                    values = [m.get(key, 0) for m in component_history]
                    if any(v > 0 for v in values):
                        has_positive_h_values = True
                        plt.semilogy(steps, values, label=key)
                    else:
                        plt.plot(steps, values, label=key)
                plt.title('Hidden Component Gradients')
                plt.xlabel('Inference Step')
                plt.ylabel('Gradient Norm')
                plt.legend()
                plt.grid(True)
            
            # Weight components
            w_keys = [k for k in component_keys if k.startswith('w_')]
            has_positive_w_values = False
            
            if w_keys:
                plt.subplot(1, 2, 2)
                for key in w_keys:
                    values = [m.get(key, 0) for m in component_history]
                    if any(v > 0 for v in values):
                        has_positive_w_values = True
                        plt.semilogy(steps, values, label=key)
                    else:
                        plt.plot(steps, values, label=key)
                plt.title('Weight Component Gradients')
                plt.xlabel('Inference Step')
                plt.ylabel('Gradient Norm')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(component_gradients_path)
            plt.close()
            component_plot_created = True
    
    # Plot error dynamics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    steps = [m['step'] for m in error_history]
    losses = [m['loss'] for m in error_history]
    recon_errors = [m['reconstruction_error'] for m in error_history]
    plt.plot(steps, losses, label='Total Loss')
    plt.plot(steps, recon_errors, label='Reconstruction Error')
    plt.title('Error Dynamics Over Steps')
    plt.xlabel('Inference Step')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Check if we have valid values for log scale
    if any(v > 0 for v in losses) or any(v > 0 for v in recon_errors):
        plt.semilogy(steps, losses, label='Total Loss')
        plt.semilogy(steps, recon_errors, label='Reconstruction Error')
    else:
        plt.plot(steps, losses, label='Total Loss')
        plt.plot(steps, recon_errors, label='Reconstruction Error')
    plt.title('Error Dynamics')
    plt.xlabel('Inference Step')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(error_dynamics_path)
    plt.close()
    
    # Plot gradient-to-energy ratio
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    h_to_w_ratio = [m['h_to_w_ratio'] for m in gradient_history]
    plt.plot(steps, h_to_w_ratio, label='Hidden/Weight Gradient Ratio')
    plt.title('Hidden to Weight Gradient Ratio')
    plt.xlabel('Inference Step')
    plt.ylabel('Ratio')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    h_max = [m['h_max_component'] for m in gradient_history] 
    w_max = [m['w_max_component'] for m in gradient_history]
    # Check if we have valid values for log scale
    if any(v > 0 for v in h_max) or any(v > 0 for v in w_max):
        plt.semilogy(steps, h_max, label='Max Hidden Component')
        plt.semilogy(steps, w_max, label='Max Weight Component')
    else:
        plt.plot(steps, h_max, label='Max Hidden Component')
        plt.plot(steps, w_max, label='Max Weight Component')
    plt.title('Maximum Component Gradients')
    plt.xlabel('Inference Step')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(gradient_ratios_path)
    plt.close()
    
    # Save histories to JSON for later analysis
    with open(f"{debug_dir}/gradient_history_epoch{epoch}.json", 'w') as f:
        json.dump(gradient_history, f, indent=2)
    with open(f"{debug_dir}/error_history_epoch{epoch}.json", 'w') as f:
        json.dump(error_history, f, indent=2)
    with open(f"{debug_dir}/component_history_epoch{epoch}.json", 'w') as f:
        json.dump(component_history, f, indent=2)
    
    # Prepare W&B logging dictionary
    log_dict = {
        'Gradients/hidden_energy': h_energy[-1],
        'Gradients/weight_energy': w_energy[-1],
        'Gradients/hidden_grad_norm': h_grad_norm[-1],
        'Gradients/weight_grad_norm': w_grad_norm[-1],
        'Gradients/h_to_w_ratio': h_to_w_ratio[-1],
        'Gradients/h_max_component': h_max[-1],
        'Gradients/w_max_component': w_max[-1],
        'Errors/total_loss': losses[-1],
        'Errors/reconstruction_error': recon_errors[-1],
        'Dynamics/gradient_plot': wandb.Image(gradient_dynamics_path),
        'Dynamics/error_plot': wandb.Image(error_dynamics_path),
        'Dynamics/ratio_plot': wandb.Image(gradient_ratios_path)
    }
    
    # Only add component plot if it was created
    if component_plot_created:
        log_dict['Dynamics/component_plot'] = wandb.Image(component_gradients_path)
    
    # Return log_dict instead of logging directly, to ensure consistent step handling
    return log_dict, loss

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

def main():
    """Main function to run the debugging process with W&B logging."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Initialize Weights & Biases
    run = wandb.init(
        entity="neural-machines",
        project="debug-transformer",
        config={
            "latent_dim": args.latent_dim,
            "num_blocks": args.num_blocks,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "inference_steps": args.inference_steps,
            "peak_lr_weights": args.peak_lr_weights,
            "peak_lr_hidden": args.peak_lr_hidden,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
            "target_class": args.target_class,
            "num_images": args.num_images
        },
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
    config = create_config(
        dataset="cifar10",
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks
    )
    
    print(f"Creating debug dataloaders for CIFAR-10...")
    train_loader, val_loader = get_debug_dataloaders(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        root_path=args.data_dir,
        train_subset_n=args.train_subset,
        test_subset_n=args.test_subset,
        target_class=args.target_class
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    print("Initializing model...")
    model = TransformerDecoder(config)
    
    # Calculate steps per epoch for learning rate schedule
    steps_per_epoch = len(train_loader)
    
    # Set up optimizers with learning rate schedule
    weights_lr_schedule = create_learning_rate_schedule(
        args.peak_lr_weights, 
        args.warmup_epochs, 
        args.epochs, 
        steps_per_epoch
    )
    
    hidden_lr_schedule = create_learning_rate_schedule(
        args.peak_lr_hidden, 
        args.warmup_epochs, 
        args.epochs, 
        steps_per_epoch
    )
    
    print(f"Using learning rate schedule with:")
    print(f"  - Peak weight LR: {args.peak_lr_weights}")
    print(f"  - Peak hidden LR: {args.peak_lr_hidden}")
    print(f"  - Warmup epochs: {args.warmup_epochs}")
    print(f"  - Weight decay: {args.weight_decay}")
    
    # Create optimizers with schedule
    optim_h = pxu.Optim(lambda: optax.sgd(hidden_lr_schedule, momentum=0.9))
    optim_w = pxu.Optim(
        lambda: optax.adamw(weights_lr_schedule, weight_decay=args.weight_decay), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    # Store validation losses
    val_losses = []
    
    # Direct inspection of a single training batch to understand gradient structure
    print("\n=== Direct Inspection of train_on_batch Output ===")
    test_batch = next(iter(train_loader.dataloader))
    x_test, _ = test_batch
    x_test = jnp.array(x_test.numpy())
    
    print("Calling train_on_batch directly to inspect outputs...")
    h_energy, w_energy, h_grad, w_grad = train_on_batch(
        1, x_test, model=model, optim_w=optim_w, optim_h=optim_h, epoch=0, step=0
    )
    
    print(f"h_energy: {h_energy}, w_energy: {w_energy}")
    print(f"h_energy type: {type(h_energy)}, w_energy type: {type(w_energy)}")
    print(f"Are energies identical? {h_energy == w_energy}")
    
    print("\nInspecting hidden gradient structure:")
    h_grad_type = type(h_grad)
    print(f"h_grad type: {h_grad_type}")
    
    if isinstance(h_grad, dict):
        print(f"h_grad keys: {h_grad.keys()}")
        for key in h_grad.keys():
            print(f"  h_grad[{key}] type: {type(h_grad[key])}")
            if isinstance(h_grad[key], dict):
                print(f"    h_grad[{key}] keys: {h_grad[key].keys()}")
    
    print("\nInspecting weight gradient structure:")
    w_grad_type = type(w_grad)
    print(f"w_grad type: {w_grad_type}")
    
    if isinstance(w_grad, dict):
        print(f"w_grad keys: {w_grad.keys()}")
        for key in w_grad.keys():
            print(f"  w_grad[{key}] type: {type(w_grad[key])}")
            if isinstance(w_grad[key], dict):
                print(f"    w_grad[{key}] keys: {w_grad[key].keys()}")
    
    print("=== End of Direct Inspection ===\n")
    
    print(f"Training for {args.epochs} epochs with W&B logging...")
    
    # # Generate initial reconstruction (epoch 0)
    # print("Generating initial reconstructions...")
    # _, _, initial_recon_path, initial_recon_logs = visualize_reconstruction(
    #     model, 
    #     optim_h, 
    #     val_loader, 
    #     T_values=[1, 2, 4, 8], 
    #     use_corruption=False,
    #     num_images=args.num_images,
    #     wandb_run=run,
    #     epoch=0
    # )
    
    # # Log initial reconstruction with step=0
    # run.log(initial_recon_logs, step=0)
    # print(f"Uploaded reconstruction image for epoch 0 to W&B")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Get current learning rates
        current_w_lr = weights_lr_schedule(epoch * steps_per_epoch)
        current_h_lr = hidden_lr_schedule(epoch * steps_per_epoch)
        print(f"Current learning rates - Weights: {current_w_lr:.6f}, Hidden: {current_h_lr:.6f}")
        
        # Train for one epoch
        train(train_loader, args.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch)
        
        # Evaluate on validation set
        val_loss = eval(train_loader, args.inference_steps, model=model, optim_h=optim_h)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.6f}")
        
        # Collect all metrics for this epoch in a single dictionary
        epoch_metrics = {
            'Losses/validation_loss': val_loss,
            'LearningRate/weights': current_w_lr,
            'LearningRate/hidden': current_h_lr
        }
        
        # Track dynamics on a sample batch
        print("Tracking batch dynamics...")
        batch = next(iter(train_loader.dataloader))
        dynamics_metrics, batch_loss = track_batch_dynamics(model, optim_h, optim_w, batch, args.inference_steps, epoch)
        epoch_metrics.update(dynamics_metrics)
        epoch_metrics['Losses/batch_loss'] = batch_loss
        print(f"Batch dynamics loss: {batch_loss:.6f}")
        
        # Generate reconstructions every 5 epochs (and for the final epoch)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Generating reconstructions for epoch {epoch+1}...")
            _, _, recon_path, recon_logs = visualize_reconstruction(
                model, 
                optim_h, 
                val_loader, 
                T_values=[1, 2, 4, 8, 16, 32], 
                use_corruption=False,
                num_images=args.num_images,
                wandb_run=run,
                epoch=epoch+1
            )
            # Add reconstruction logs to the epoch metrics
            epoch_metrics.update(recon_logs)
        
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
    
    # # Final model evaluation is now redundant since we're creating reconstructions during training
    # # But we'll keep it for backward compatibility and add it to the summary
    # print("Final model evaluation for summary...")
    # _, _, final_recon_path, _ = visualize_reconstruction(
    #     model, 
    #     optim_h, 
    #     val_loader, 
    #     T_values=[1, 2, 4, 8], 
    #     use_corruption=False,
    #     num_images=args.num_images,
    #     wandb_run=run,
    #     epoch="final"
    # )
    
    # # Save final reconstruction path to run summary
    # if final_recon_path:
    #     wandb.summary["final_reconstruction"] = wandb.Image(final_recon_path)
    
    # Close W&B after all logging is complete
    wandb.finish()
    
    # Plot validation loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), val_losses, marker='o')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    os.makedirs("../debug_logs", exist_ok=True)
    plt.savefig("../debug_logs/validation_loss_curve.png")
    
    print("Debug run completed!")

if __name__ == "__main__":
    main() 