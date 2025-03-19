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
import pcx.utils as pxu
import pcx.nn as pxnn
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
    visualize_reconstruction
)

# Import the dataloader utility from utils_dataloader
from src.utils_dataloader import get_dataloaders, TorchDataloader

# Set up basic parameters - pushing the model capacity higher
BATCH_SIZE = 128  # Even larger batch size
LATENT_DIM = 512  # Larger latent dimension
NUM_EPOCHS = 20   # More epochs for convergence
NUM_BLOCKS = 6    # More transformer blocks for capacity
INFERENCE_STEPS = 8  # More inference steps

# Learning rates with schedule parameters
PEAK_LR_WEIGHTS = 1e-3  # Higher peak learning rate
PEAK_LR_HIDDEN = 0.01   # Higher peak learning rate
WEIGHT_DECAY = 2e-4     # Slightly stronger weight decay
WARMUP_EPOCHS = 5       # Warmup period

# Dataset parameters - using more data
TRAIN_SUBSET = None  # Use the full training set (50,000 images)
TEST_SUBSET = 5000   # Larger test set
TARGET_CLASS = None  # Use all classes

def create_config(dataset="cifar10", latent_dim=LATENT_DIM, num_blocks=NUM_BLOCKS):
    """Create a TransformerConfig based on the dataset name."""
    if dataset == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=384,  # Larger hidden size
            num_heads=12,     # More attention heads
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
    parser = argparse.ArgumentParser(description='Train a jflux-inspired transformer model for predictive coding')
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
                        help='Number of samples to use from the training set (default: all)')
    parser.add_argument('--test-subset', type=int, default=TEST_SUBSET,
                        help='Number of samples to use from the test set (default: all)')
    parser.add_argument('--target-class', type=int, default=TARGET_CLASS,
                        help='Filter the dataset to a specific class (0-9 for CIFAR-10) (default: all classes)')
    return parser.parse_args()


def get_advanced_dataloaders(dataset_name, batch_size, root_path, train_subset_n=None, test_subset_n=None, target_class=None):
    """Get data loaders with advanced augmentation for better training."""
    
    # More aggressive data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Add rotation augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset with augmentation
    dataset_root = root_path + "cifar10/"
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=train_transform,
        download=True,
        train=True,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=test_transform,
        download=False,
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
    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    return train_dataloader, test_dataloader


def create_cosine_lr_schedule(peak_lr, total_epochs, warmup_epochs, steps_per_epoch):
    """Create a learning rate schedule with warmup and cosine decay."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def schedule_fn(step):
        step = step % total_steps  # Reset after full schedule
        
        # Convert to JAX arrays for compatibility
        step_f = jnp.array(step, dtype=jnp.float32)
        warmup_steps_f = jnp.array(warmup_steps, dtype=jnp.float32)
        total_steps_f = jnp.array(total_steps, dtype=jnp.float32)
        
        # Linear warmup phase
        warmup_lr = peak_lr * (step_f / warmup_steps_f)
        
        # Cosine decay phase
        decay_fraction = (step_f - warmup_steps_f) / (total_steps_f - warmup_steps_f)
        cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_fraction))
        cosine_lr = peak_lr * cosine_decay
        
        # Return warmup_lr during warmup, cosine_lr afterward
        return jnp.where(step_f < warmup_steps_f, warmup_lr, cosine_lr)
    
    return schedule_fn


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration for CIFAR-10
    config = create_config(
        dataset="cifar10",
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks
    )
    
    print(f"Creating dataloaders for CIFAR-10 with advanced augmentation...")
    
    # Use the custom function with augmentation
    train_loader, val_loader = get_advanced_dataloaders(
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
    if args.target_class is not None:
        print(f"Using only samples from class {args.target_class}")
    
    print("Initializing model...")
    # Create model with configuration
    model = TransformerDecoder(config)
    
    # Set up learning rate schedules
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    # Calculate learning rate schedules
    lr_weights_fn = create_cosine_lr_schedule(PEAK_LR_WEIGHTS, args.epochs, WARMUP_EPOCHS, steps_per_epoch)
    lr_hidden_fn = create_cosine_lr_schedule(PEAK_LR_HIDDEN, args.epochs, WARMUP_EPOCHS, steps_per_epoch)
    
    # We need to track the step count for the learning rate schedule
    # Using a counter wrapped in a list to make it mutable through a closure
    step_counter = [0]
    
    def get_lr_weights():
        step = step_counter[0]
        step_counter[0] += 1
        return lr_weights_fn(step)
    
    def get_lr_hidden():
        step = step_counter[0] - 1  # Already incremented by weights
        return lr_hidden_fn(step)
    
    print(f"Using learning rate schedule with peak values: LR_WEIGHTS={PEAK_LR_WEIGHTS}, LR_HIDDEN={PEAK_LR_HIDDEN}")
    print(f"Warmup for {WARMUP_EPOCHS} epochs, then cosine decay")
    
    # Set up the optimizers with learning rate schedules
    optim_h = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(PEAK_LR_HIDDEN, momentum=0.9)  # Using fixed LR since schedule is tricky with JAX
        ))
    optim_w = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(PEAK_LR_WEIGHTS, weight_decay=WEIGHT_DECAY, b1=0.9, b2=0.999)
        ), pxu.M(pxnn.LayerParam)(model))
    
    # Store validation losses to track progress
    val_losses = []
    best_val_loss = float('inf')
    
    # Create results directory
    results_dir = "../results/advanced_transformer"
    os.makedirs(results_dir, exist_ok=True)
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Implement manual learning rate adjustment
        if epoch < WARMUP_EPOCHS:
            # Warmup phase
            lr_scale = (epoch + 1) / WARMUP_EPOCHS
            current_lr_weights = PEAK_LR_WEIGHTS * lr_scale
            current_lr_hidden = PEAK_LR_HIDDEN * lr_scale
        else:
            # Cosine decay phase
            decay_fraction = (epoch + 1 - WARMUP_EPOCHS) / (args.epochs - WARMUP_EPOCHS)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * decay_fraction))
            current_lr_weights = PEAK_LR_WEIGHTS * cosine_factor
            current_lr_hidden = PEAK_LR_HIDDEN * cosine_factor
        
        # Create new optimizers with the current learning rates
        optim_h = pxu.Optim(lambda: optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.sgd(current_lr_hidden, momentum=0.9)
            ))
        optim_w = pxu.Optim(lambda: optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(current_lr_weights, weight_decay=WEIGHT_DECAY, b1=0.9, b2=0.999)
            ), pxu.M(pxnn.LayerParam)(model))
        
        print(f"Current learning rates: LR_WEIGHTS={current_lr_weights:.6f}, LR_HIDDEN={current_lr_hidden:.6f}")
        
        # Train for one epoch
        train(train_loader, args.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h)
        
        # Evaluate on validation set
        val_loss = eval(val_loader, args.inference_steps, model=model, optim_h=optim_h)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.6f}")
        
        # Print some stats to help monitor convergence
        if len(val_losses) > 1:
            loss_diff = val_losses[-1] - val_losses[-2]
            percent_change = (loss_diff / val_losses[-2]) * 100
            print(f"Loss change: {loss_diff:.6f} ({percent_change:.2f}%)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Create a plot of the validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epoch + 2), val_losses, marker='o')
            plt.title('Validation Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.grid(True)
            plt.savefig(f"{results_dir}/validation_loss_curve_epoch_{epoch+1}.png")
            plt.close()
            
            # Visualize reconstruction
            visualize_reconstruction(
                model, 
                optim_h, 
                val_loader, 
                T_values=[1, 4, 8], 
                use_corruption=False,
                num_images=4
            )
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")
        
        # Check if we've reached CNN-level performance
        if val_loss <= 0.19:
            print(f"Reached target validation loss of 0.19 or better: {val_loss:.6f}")
            print(f"Stopping training early at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {val_losses.index(min(val_losses))+1})")
    print(f"Total loss reduction: {(val_losses[0] - min(val_losses)) / val_losses[0] * 100:.2f}%")
    
    # Create a plot of the validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig(f"{results_dir}/final_validation_loss_curve.png")
    
    # Visualize results with different inference steps
    print("Visualizing final reconstruction with different inference steps...")
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 4, 8], 
        use_corruption=False,
        num_images=4
    )
    
    # Visualize inpainting with different inference steps
    print("Visualizing final inpainting with different inference steps...")
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 4, 8], 
        use_corruption=True,
        corrupt_ratio=0.5,
        num_images=4
    )

if __name__ == "__main__":
    main() 