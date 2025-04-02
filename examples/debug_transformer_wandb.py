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

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import the TransformerDecoder and utility functions
from src.decoder_transformer import (
    TransformerDecoder, 
    TransformerConfig,
    train, 
    eval, 
    unmask_on_batch,
    forward
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
    reconstruction_every_n_epochs: int = 1
    validation_every_n_epochs: int = 1
    use_corruption: bool = True
    corrupt_ratio: float = 0.5

    # Visualization settings
    num_images: int = 5
    
    # Model architecture
    hidden_size: int = 48
    num_heads: int = 6
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    patch_size: int = 4
    axes_dim: List[int] = field(default_factory=lambda: [16, 16])
    theta: int = 10_000
    
    # Training settings
    use_noise: bool = True
    batch_size: int = 5
    epochs: int = 10
    inference_steps: int = 100
    eval_inference_steps: int = 100
    reconstruction_steps: List[int] = field(default_factory=lambda: [10, 200, 400])
    peak_lr_weights: float = 1e-3
    peak_lr_hidden: float = 0.01
    weight_decay: float = 2e-4
    warmup_epochs: int = 1
    use_lr_schedule: bool = False
    seed: int = 42
    
    # Early stopping settings
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.00001

# Predefined configurations for easy experimentation
MODEL_CONFIGS = {
    "debug_tiny": ModelConfig(
        name="debug_tiny",
        hidden_size=128,
        num_heads=4,
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
                 mlp_ratio=4.0, patch_size=4, axes_dim=None, theta=10_000, use_noise=True):
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
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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

def visualize_reconstruction(model, optim_h, dataloader, T_values=[24], use_corruption=False, corrupt_ratio=0.5, target_class=None, num_images=2, wandb_run=None, epoch=None, step=None):
    
    # Import or define STATUS_FORWARD constant
    STATUS_FORWARD = "forward"
    
    # Extract image shape statically
    image_shape = model.config.image_shape
    num_channels = image_shape[0]
    
    orig_images = []
    recon_images = {T: [] for T in T_values}
    labels_list = []
    dataloader_iter = iter(dataloader)
    
    # Get the single batch
    try:
        x, label = next(dataloader_iter)
        
        x = jnp.array(x)
        
        # Process each image in the batch separately
        for i in range(num_images):
            # Get single image and reshape
            single_x = x[i:i+1]  # Keep batch dimension but with size 1
            orig_images.append(jnp.reshape(single_x[0], image_shape))
            
            # Handle label
            if hasattr(label, 'item'):
                labels_list.append(label[i].item() if len(label.shape) > 0 else label.item())
            else:
                labels_list.append(None)
                
            for T in T_values:
                # Process single image
                loss, x_hat = unmask_on_batch(
                    use_corruption=use_corruption, 
                    corrupt_ratio=corrupt_ratio, 
                    T=T, 
                    x=single_x,  # Pass single image
                    model=model, 
                    optim_h=optim_h
                )
                
                x_hat_single = jnp.reshape(x_hat[0], image_shape)
                recon_images[T].append(x_hat_single)
                
    except StopIteration:
        print("Warning: No data available in dataloader")
        return [], {}
    
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
    
    # Add debug information about the structure
    print(f"h_grad type: {type(h_grad)}")
    if isinstance(h_grad, dict) and 'model' in h_grad:
        print(f"model_grads type: {type(h_grad['model'])}")
        if hasattr(h_grad['model'], 'vodes'):
            print(f"vodes type: {type(h_grad['model'].vodes)}")
            print(f"Number of vodes: {len(h_grad['model'].vodes)}")
            if len(h_grad['model'].vodes) > 0:
                first_vode = h_grad['model'].vodes[0]
                print(f"First vode type: {type(first_vode)}")
                if hasattr(first_vode, 'h'):
                    print(f"First vode.h type: {type(first_vode.h)}")
                    if hasattr(first_vode.h, 'value'):
                        print(f"First vode.h.value type: {type(first_vode.h.value)}")
                        print(f"First vode.h.value shape: {first_vode.h.value.shape}")
    
    try:
        # Access the model gradients
        model_grads = h_grad['model']
        
        # Access the vodes list
        vodes = model_grads.vodes
        
        # For each vode, extract the gradient tensor and calculate its L2 norm
        for i in range(len(vodes)):
            vode = vodes[i]
            h_grad_tensor = vode.h
            print("h_grad_tensor", h_grad_tensor)
            
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
                print(f"Vode {i} L2 norm: {l2_norm}")
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
        x = row[0]      # epoch
        series = row[1]  # vode_index
        y = row[2]      # energy/grad_norm
        
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
                series_col: f"Vode {series}"
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
        mode="offline"  # Change to online for immediate verification
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
        hidden_size=config.hidden_size,
        num_blocks=config.num_blocks,
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
    
    # Store energy and gradient data for all epochs
    all_epochs_energy_data = []
    all_epochs_grad_data = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopped = False
    
    print(f"Training for {config.epochs} epochs with W&B logging...")

    # Initialize the model (set h values of the Vodes)
    for x, _ in train_loader:
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(x.numpy(), model=model)
    
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
        
        # Create logs and evaluate on validation set every N epochs (and for the final epoch)
        if (epoch + 1) % config.validation_every_n_epochs == 0 or epoch == config.epochs - 1 or (config.use_early_stopping and early_stopped and epoch == early_stopped_epoch):
            # Evaluate on validation set
            val_loss = eval(train_loader, config.eval_inference_steps, model=model, optim_h=optim_h)
            val_losses.append(val_loss)
            print(f"Validation loss: {val_loss:.6f}")
            
            # Collect all metrics for this epoch in a single dictionary
            epoch_metrics = {
                'Losses/validation_loss': val_loss,
                'LearningRate/weights': current_w_lr,
                'LearningRate/hidden': current_h_lr
            }
        
        # # Generate reconstructions every N epochs (and for the final epoch)
        # if (epoch + 1) % config.reconstruction_every_n_epochs == 0 or epoch == config.epochs - 1 or (config.use_early_stopping and early_stopped and epoch == early_stopped_epoch):
            
            # # Log detailed vode statistics and get processed data for summary
            # processed_energy_data, processed_grad_data = log_vode_stats(model, h_grad, w_grad, run, epoch)
            
            # # Add the processed data to our summary collections
            # all_epochs_energy_data.extend(processed_energy_data)
            # all_epochs_grad_data.extend(processed_grad_data)
            
            print(f"Generating reconstructions for epoch {epoch+1}...")
            _, _, recon_path, recon_logs = visualize_reconstruction(
                model, 
                optim_h, 
                train_loader, 
                T_values=config.reconstruction_steps, 
                use_corruption=config.use_corruption,
                corrupt_ratio=config.corrupt_ratio,
                num_images=config.num_images,
                wandb_run=run,
                epoch=epoch+1
            )
            # Add reconstruction logs to the epoch metrics
            epoch_metrics.update(recon_logs)
        
            # # Add system metrics
            # epoch_metrics.update({
            #     'System/GPU_Memory_Allocated': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            #     'System/GPU_Memory_Cached': torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
            #     'System/CPU_Memory_Usage': psutil.Process().memory_info().rss / 1024**2
            # })

            # Log all metrics for this epoch with a consistent step number
            run.log(epoch_metrics, step=epoch+1)  # Use epoch+1 to start from step 1

            # Early stopping check
            if config.use_early_stopping:
                if val_loss < (best_val_loss - config.early_stopping_min_delta):
                    # Found a better model
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save the best model here if desired
                    # Could implement model state saving logic here
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.6f})")
                    
                    if epochs_without_improvement >= config.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs!")
                        early_stopped = True
                        early_stopped_epoch = epoch
                        break
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")
    
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

if __name__ == "__main__":
    main() 