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
    model_debugger,  # Import the debugger
    forward,
    train_on_batch  # Add train_on_batch import
)

# Set up basic parameters - pushing the model capacity higher
BATCH_SIZE = 1  # Even larger batch size
LATENT_DIM = 128  # Larger latent dimension
NUM_EPOCHS = 10   # More epochs for convergence
NUM_BLOCKS = 1    # More transformer blocks for capacity
INFERENCE_STEPS = 32  # More inference steps

# Learning rates with schedule parameters
PEAK_LR_WEIGHTS = 1e-3  # Higher peak learning rate
PEAK_LR_HIDDEN = 0.01   # Higher peak learning rate
WEIGHT_DECAY = 2e-4     # Slightly stronger weight decay
WARMUP_EPOCHS = 5       # Warmup period

# Dataset parameters - using more data
TRAIN_SUBSET = 1  # Use the full training set (50,000 images)
TEST_SUBSET = 1   # Larger test set
TARGET_CLASS = None  # Use all classes


def create_config(dataset="cifar10", latent_dim=LATENT_DIM, num_blocks=NUM_BLOCKS):
    """Create a TransformerConfig based on the dataset name."""
    if dataset == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=48,  # Larger hidden size
            num_heads=6,     # More attention heads
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
    parser = argparse.ArgumentParser(description='Debug a transformer model for predictive coding')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM,
                        help=f'Latent dimension size (default: {LATENT_DIM})')
    parser.add_argument('--num-blocks', type=int, default=NUM_BLOCKS,
                        help=f'Number of transformer blocks (default: {NUM_BLOCKS})')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--inference-steps', type=int, default=INFERENCE_STEPS,
                        help=f'Number of inference steps (default: {INFERENCE_STEPS})')
    parser.add_argument('--data-dir', type=str, default='../datasets/',
                        help='Directory to store datasets (default: ../datasets/)')
    parser.add_argument('--train-subset', type=int, default=TRAIN_SUBSET,
                        help='Number of samples to use from the training set (default: 100)')
    parser.add_argument('--test-subset', type=int, default=TEST_SUBSET,
                        help='Number of samples to use from the test set (default: 50)')
    parser.add_argument('--target-class', type=int, default=TARGET_CLASS,
                        help='Filter the dataset to a specific class (0-9 for CIFAR-10) (default: all classes)')
    parser.add_argument('--debug-interval', type=int, default=1,
                        help='How often to log detailed debugging info (in epochs)')
    parser.add_argument('--peak-lr-weights', type=float, default=PEAK_LR_WEIGHTS,
                        help=f'Peak learning rate for weights (default: {PEAK_LR_WEIGHTS})')
    parser.add_argument('--peak-lr-hidden', type=float, default=PEAK_LR_HIDDEN,
                        help=f'Peak learning rate for hidden states (default: {PEAK_LR_HIDDEN})')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help=f'Weight decay for AdamW optimizer (default: {WEIGHT_DECAY})')
    parser.add_argument('--warmup-epochs', type=int, default=WARMUP_EPOCHS,
                        help=f'Number of warmup epochs for learning rate (default: {WARMUP_EPOCHS})')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

# Add a learning rate schedule function
def create_learning_rate_schedule(base_lr, warmup_epochs, total_epochs, steps_per_epoch):
    """Create a learning rate schedule with warmup and cosine decay."""
    def lr_schedule(step):
        # Convert step to epoch (as a scalar value)
        epoch = step / steps_per_epoch
        
        # Use jnp.where instead of if/else for JAX tracing compatibility
        warmup_lr = base_lr * (epoch / warmup_epochs)
        
        # Cosine decay calculation
        decay_ratio = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        decay_lr = base_lr * cosine_factor
        
        # Use jnp.where for conditional: warmup_lr if epoch < warmup_epochs else decay_lr
        return jnp.where(epoch < warmup_epochs, warmup_lr, decay_lr)
    
    return lr_schedule

def get_debug_dataloaders(dataset_name, batch_size, root_path, train_subset_n=None, test_subset_n=None, target_class=None):
    """Get data loaders with simple augmentation for debugging."""
    
    # Simple augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset with augmentation
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
    
    # Filter by target class if specified
    if target_class is not None:
        target_indices = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        train_dataset = Subset(train_dataset, target_indices)
        target_indices = (test_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        test_dataset = Subset(test_dataset, target_indices)
    
    # Optionally restrict the datasets further
    if train_subset_n is not None:
        all_idx = list(range(len(train_dataset)))
        train_dataset = Subset(train_dataset, all_idx[:train_subset_n])
    if test_subset_n is not None:
        all_idx = list(range(len(test_dataset)))
        test_dataset = Subset(test_dataset, all_idx[:test_subset_n])
    
    # Create dataloaders
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
    
    # Create wrapper class to match the expected interface
    class TorchDataloader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
        
        def __iter__(self):
            return iter(self.dataloader)
        
        def __len__(self):
            return len(self.dataloader)
    
    return TorchDataloader(train_dataloader), TorchDataloader(test_dataloader)

def debug_plot_unpatchify_test(model, patch_size=4):
    """Test unpatchify and patchify operations with known data to verify correctness."""
    # Create debug directory
    debug_dir = "../debug_logs/unpatchify_test"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create a simple test pattern
    c, h, w = model.config.image_shape
    test_image = np.zeros((c, h, w))
    
    # Create a gradient pattern
    for i in range(h):
        for j in range(w):
            for k in range(c):
                test_image[k, i, j] = (i / h) * (j / w) * (k + 1) / c
    
    # Add some distinctive patterns
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Add a small square in each patch corner
            test_image[0, i, j] = 1.0  # Red corner
            test_image[1, i, j+patch_size-1] = 1.0  # Green corner
            test_image[2, i+patch_size-1, j] = 1.0  # Blue corner
    
    # Convert to JAX array
    test_image = jnp.array(test_image)
    
    # Patchify
    patches = model._patchify(test_image)
    print(f"Patches shape: {patches.shape}")
    
    # Unpatchify back to image
    reconstructed = model._unpatchify(patches)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(np.array(test_image), (1, 2, 0)))
    plt.title("Original Test Pattern")
    
    # Patches visualization
    plt.subplot(1, 3, 2)
    num_patches = patches.shape[0]
    side = int(np.sqrt(num_patches))
    patch_dim = patches.shape[1]
    patch_viz = patches.reshape(side, side, patch_dim)
    patch_viz = np.mean(patch_viz, axis=-1)  # Average over patch dimension
    plt.imshow(patch_viz, cmap='viridis')
    plt.title("Patches (Mean Value)")
    
    # Reconstructed
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(np.array(reconstructed), (1, 2, 0)))
    plt.title("Reconstructed")
    
    plt.tight_layout()
    plt.savefig(f"{debug_dir}/unpatchify_test.png")
    plt.close()
    
    # Check if reconstruction is identical to original
    mse = jnp.mean(jnp.square(test_image - reconstructed))
    print(f"Reconstruction MSE: {mse}")
    
    return mse

def inspect_model_parameters(model):
    """Log information about model parameters to help with debugging."""
    parameter_info = {}
    
    # Function to recursively inspect a tree of parameters
    def inspect_tree(tree, prefix=""):
        nonlocal parameter_info
        
        if hasattr(tree, '__dict__'):
            for name, value in tree.__dict__.items():
                # Skip private attributes
                if name.startswith('_'):
                    continue
                    
                full_name = f"{prefix}.{name}" if prefix else name
                
                # If it's a leaf node with a value, log its information
                if hasattr(value, 'shape') and isinstance(value.shape, tuple):
                    param_shape = value.shape
                    param_info = {
                        'shape': str(param_shape),
                        'size': np.prod(param_shape),
                        'min': float(jnp.min(value)) if hasattr(value, 'min') else None,
                        'max': float(jnp.max(value)) if hasattr(value, 'max') else None,
                        'mean': float(jnp.mean(value)) if hasattr(value, 'mean') else None,
                        'std': float(jnp.std(value)) if hasattr(value, 'std') else None
                    }
                    parameter_info[full_name] = param_info
                # For non-leaf nodes, recurse
                else:
                    inspect_tree(value, full_name)
    
    # Start inspection from the model
    try:
        inspect_tree(model)
        
        # Save parameter information
        debug_dir = "../debug_logs"
        os.makedirs(debug_dir, exist_ok=True)
        
        import json
        with open(os.path.join(debug_dir, 'model_parameters.json'), 'w') as f:
            json.dump(parameter_info, f, indent=2)
            
        # Also print summary
        total_params = sum(info['size'] for info in parameter_info.values() if info['size'] is not None)
        print(f"Model has {len(parameter_info)} parameter groups with approximately {total_params:,} total parameters")
        
    except Exception as e:
        print(f"Error inspecting model parameters: {e}")

def debug_one_batch(model, optim_h, optim_w, batch, inference_steps, epoch):
    """Run detailed debugging on a single batch without causing JAX tracing errors."""
    x, y = batch
    x_numpy = x.numpy()
    x = jnp.array(x_numpy)
    
    # Enable more verbose logging for this batch
    old_logging = model_debugger.enable_logging
    model_debugger.enable_logging = True
    
    # Create a non-JIT version of forward pass for debugging only
    def debug_forward(x_input):
        # Run a standard forward pass for debugging purposes
        with pxu.step(model, "forward", clear_params=pxc.VodeParam.Cache):
            z = forward(x_input, model=model)
            return z
    
    # First do a normal training step
    h_energy, w_energy, h_grad, w_grad = train_on_batch(
        inference_steps, x, model=model, optim_w=optim_w, optim_h=optim_h, epoch=epoch, step=0
    )
    
    # Explicitly log energy and gradient information right after training
    try:
        print("Logging energy and gradient information...")
        model_debugger.log_energy(0, h_energy, w_energy)
        model_debugger.log_gradient_norms(0, h_grad, w_grad)
        
        # Manually flatten gradients for visualization
        h_grad_flat = model_debugger._flatten_gradients(h_grad.get("model", {}))
        w_grad_flat = model_debugger._flatten_gradients(w_grad.get("model", {}))
        
        if h_grad_flat is not None and w_grad_flat is not None:
            # Log gradient statistics
            h_grad_np = np.array(h_grad_flat)
            w_grad_np = np.array(w_grad_flat)
            
            # Create gradient visualizations
            debug_dir = "../debug_logs/gradients"
            os.makedirs(debug_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(h_grad_np.flatten(), bins=50)
            plt.title("Hidden Gradients Distribution")
            
            plt.subplot(1, 2, 2)
            plt.hist(w_grad_np.flatten(), bins=50)
            plt.title("Weight Gradients Distribution")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plt.savefig(f"{debug_dir}/gradient_hist_epoch{epoch}_{timestamp}.png")
            plt.close()
            
            # Calculate and log gradient statistics
            h_grad_stats = {
                'min': float(np.min(h_grad_np)),
                'max': float(np.max(h_grad_np)),
                'mean': float(np.mean(h_grad_np)),
                'std': float(np.std(h_grad_np)),
                'abs_mean': float(np.mean(np.abs(h_grad_np))),
                'norm': float(np.linalg.norm(h_grad_np))
            }
            
            w_grad_stats = {
                'min': float(np.min(w_grad_np)),
                'max': float(np.max(w_grad_np)),
                'mean': float(np.mean(w_grad_np)),
                'std': float(np.std(w_grad_np)),
                'abs_mean': float(np.mean(np.abs(w_grad_np))),
                'norm': float(np.linalg.norm(w_grad_np))
            }
            
            # Save gradient statistics
            import json
            with open(os.path.join(debug_dir, f'gradient_stats_epoch{epoch}.json'), 'w') as f:
                json.dump({'hidden_gradients': h_grad_stats, 'weight_gradients': w_grad_stats}, f, indent=2)
            
            print(f"Hidden gradients - min: {h_grad_stats['min']:.6f}, max: {h_grad_stats['max']:.6f}, mean: {h_grad_stats['mean']:.6f}, norm: {h_grad_stats['norm']:.6f}")
            print(f"Weight gradients - min: {w_grad_stats['min']:.6f}, max: {w_grad_stats['max']:.6f}, mean: {w_grad_stats['mean']:.6f}, norm: {w_grad_stats['norm']:.6f}")
    except Exception as e:
        print(f"Error logging gradient information: {e}")
    
    # Then do debugging operations separately (outside of JIT context)
    try:
        print("Collecting debugging information...")
        # Log input statistics
        model_debugger.log_activation_stats('input_batch', x)
        
        # Extract and log block outputs manually
        for i, block in enumerate(model.transformer_blocks):
            if hasattr(block, 'module'):
                # Get parameters from the transformer block
                try:
                    if hasattr(block.module, 'parameters'):
                        params = block.module.parameters()
                        for param_name, param in params.items():
                            model_debugger.log_activation_stats(f'block_{i}_{param_name}', param)
                except Exception as e:
                    print(f"Error logging block {i} parameters: {e}")
    except Exception as e:
        print(f"Error during debug statistics collection: {e}")

    # Evaluate reconstruction quality
    loss, x_hat = eval_on_batch_partial(
        use_corruption=False, corrupt_ratio=0.5, T=inference_steps, 
        x=x, model=model, optim_h=optim_h
    )
    
    # Create detailed visualizations
    debug_dir = "../debug_logs/batch_debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Safely visualize original vs reconstruction for this batch
    try:
        plt.figure(figsize=(10, 5))
        for i in range(min(4, x.shape[0])):
            # Original
            plt.subplot(2, 4, i+1)
            
            # Safely handle original images
            img = x_numpy[i]
            if img.shape[0] == 3:  # If channels-first format (C, H, W)
                plt.imshow(np.transpose(img, (1, 2, 0)))
            else:  # If already in (H, W, C) format or grayscale
                plt.imshow(img)
            plt.title(f"Original {i}")
            plt.axis('off')
            
            # Reconstructed - safely handle different formats
            plt.subplot(2, 4, i+5)
            try:
                x_hat_np = np.array(x_hat[1][i])
                # Check dimensions before transpose
                if x_hat_np.shape[0] == 3 and len(x_hat_np.shape) == 3:  # If channels-first (C, H, W)
                    plt.imshow(np.transpose(x_hat_np, (1, 2, 0)))
                else:  # If already in (H, W, C) or grayscale
                    plt.imshow(x_hat_np)
                plt.title(f"Recon {i} (loss={loss:.4f})")
            except Exception as e:
                print(f"Error displaying reconstruction {i}: {e}")
                plt.text(0.5, 0.5, "Error displaying image", 
                         horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/batch_recon_epoch{epoch}_{timestamp}.png")
        plt.close()
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Debug activation histograms
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(x_numpy.flatten(), bins=50)
        plt.title("Input Histogram")
        
        try:
            plt.subplot(2, 2, 2)
            output_np = np.array(x_hat[1]).flatten()
            plt.hist(output_np, bins=50)
            plt.title("Output Histogram")
            
            # Add input-output difference histogram
            plt.subplot(2, 2, 3)
            # Ensure shapes match before subtraction
            if np.array(x_hat[1]).shape == x_numpy.shape:
                diff = np.array(x_hat[1]) - x_numpy
                plt.hist(diff.flatten(), bins=50)
                plt.title("Input-Output Difference")
            else:
                plt.text(0.5, 0.5, f"Shape mismatch: {np.array(x_hat[1]).shape} vs {x_numpy.shape}", 
                         horizontalalignment='center', verticalalignment='center')
        except Exception as e:
            print(f"Error plotting output histograms: {e}")
        
        # Add histogram of gradient norms if available
        if hasattr(model_debugger, 'gradient_norm_history') and model_debugger.gradient_norm_history:
            plt.subplot(2, 2, 4)
            grad_norms = [entry['w_grad_norm'] for entry in model_debugger.gradient_norm_history 
                        if entry['w_grad_norm'] is not None]
            if grad_norms:
                plt.hist(grad_norms, bins=20)
                plt.title("Weight Gradient Norm History")
                plt.xlabel("Gradient Norm")
                plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/histograms_epoch{epoch}_{timestamp}.png")
        plt.close()
    except Exception as e:
        print(f"Error creating histograms: {e}")
    
    # Test patchify/unpatchify on this batch
    try:
        # Get a single image from the batch
        single_img = x[0]
        
        # Patchify
        patches = model._patchify(single_img)
        
        # Debug patch values
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.hist(np.array(patches).flatten(), bins=50)
        plt.title("Patch Values Histogram")
        
        plt.subplot(2, 2, 2)
        # Visualize first few patches
        num_patches_to_show = min(64, patches.shape[0])
        patch_viz = np.array(patches[:num_patches_to_show])
        plt.imshow(patch_viz, aspect='auto')
        plt.title(f"First {num_patches_to_show} Patches")
        plt.colorbar()
        
        # Unpatchify
        reconstructed = model._unpatchify(patches)
        
        # Compare original and reconstructed
        plt.subplot(2, 2, 3)
        if x_numpy[0].shape[0] == 3:  # Check if channels-first (C, H, W)
            plt.imshow(np.transpose(x_numpy[0], (1, 2, 0)))
        else:
            plt.imshow(x_numpy[0])
        plt.title("Original Image")
        
        plt.subplot(2, 2, 4)
        recon_np = np.array(reconstructed)
        if recon_np.shape[0] == 3:  # Check if channels-first (C, H, W)
            plt.imshow(np.transpose(recon_np, (1, 2, 0)))
        else:
            plt.imshow(recon_np)
        plt.title("Patchify->Unpatchify")
        
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/patch_debug_epoch{epoch}_{timestamp}.png")
        plt.close()
    except Exception as e:
        print(f"Error during patch debugging: {e}")
    
    # Force save all logs immediately to ensure they're written to disk
    model_debugger.save_all_logs()
    
    # Restore previous logging state
    model_debugger.enable_logging = old_logging
    
    return loss

def main():
    """Main function to run the debugging process."""
    print(jax.devices())
    print(jax.device_count())
# Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    print(f"Creating configuration for debugging CIFAR-10 transformer...")
    
    # Create configuration for CIFAR-10
    config = create_config(
        dataset="cifar10",
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks
    )
    
    print(f"Creating debug dataloaders for CIFAR-10...")
    
    # Use simple dataloaders for debugging
    train_loader, val_loader = get_debug_dataloaders(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        root_path=args.data_dir,
        train_subset_n=args.train_subset,
        test_subset_n=args.test_subset,
        target_class=args.target_class
    )
    
    # Print information about the dataset
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    print("Initializing model...")
    # Create model with configuration
    model = TransformerDecoder(config)
    
    # Inspect model parameters
    inspect_model_parameters(model)
    
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
    
    # Print learning rate schedule information
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
    
    # Store validation losses to track progress
    val_losses = []

    # Validate image shape
    print(f"Image shape: {train_loader.dataloader.dataset[0][0].shape}")
    
    # Train and debug the model
    print(f"Training for {args.epochs} epochs with debugging...")

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
        
        # # Print loss change if not first epoch
        # if epoch > 0:
        #     loss_diff = val_losses[-1] - val_losses[-2]
        #     percent_change = (loss_diff / val_losses[-2]) * 100
        #     print(f"Loss change: {loss_diff:.6f} ({percent_change:.2f}%)")
        
        # # Run detailed debugging at specified intervals
        # if epoch % args.debug_interval == 0:
        #     print("Running detailed debugging on a batch...")
        #     # Get a single batch for debugging
        #     batch = next(iter(train_loader.dataloader))
        #     batch_loss = debug_one_batch(model, optim_h, optim_w, batch, args.inference_steps, epoch)
        #     print(f"Debug batch loss: {batch_loss:.6f}")
            
        #     # Visualize reconstructions
        #     print("Generating reconstruction visualizations...")
        #     visualize_reconstruction(
        #         model, 
        #         optim_h, 
        #         val_loader, 
        #         T_values=[1, args.inference_steps], 
        #         use_corruption=False,
        #         num_images=4
        #     )
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")

    print("Final model evaluation...")
    # Create a final visualization with different inference steps
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 2, 4, 8, 16, 32, 64], 
        use_corruption=False,
        num_images=1
    )
    
    # # Test inpainting with corruption
    # visualize_reconstruction(
    #     model, 
    #     optim_h, 
    #     val_loader, 
    #     T_values=[1, 2, 4, 8], 
    #     use_corruption=True,
    #     corrupt_ratio=0.5,
    #     num_images=4
    # )
    
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