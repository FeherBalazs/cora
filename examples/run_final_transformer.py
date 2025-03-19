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

# Set up basic parameters based on our successful experiment
BATCH_SIZE = 64  # Larger batch size works well
LATENT_DIM = 256
NUM_EPOCHS = 25  # Run for a bit longer for better convergence
NUM_BLOCKS = 4   # Increased to 4 transformer blocks
INFERENCE_STEPS = 4

# Learning rates - the higher learning rates with data augmentation worked well
INITIAL_LR_WEIGHTS = 5e-4
INITIAL_LR_HIDDEN = 0.005
WEIGHT_DECAY = 1e-4

# Dataset parameters
TRAIN_SUBSET = 5000  # Using more training data for final model
TEST_SUBSET = 1000
TARGET_CLASS = None

def create_config(dataset="cifar10", latent_dim=LATENT_DIM, num_blocks=NUM_BLOCKS):
    """Create a TransformerConfig based on the dataset name."""
    if dataset == "cifar10":
        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=256,
            num_heads=8,
            num_blocks=num_blocks,
            mlp_ratio=4.0,
            patch_size=4,
            axes_dim=[16, 16],
            theta=10_000,
            use_noise=True  # Enable noise for better initialization
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


def get_augmented_dataloaders(dataset_name, batch_size, root_path, train_subset_n=None, test_subset_n=None, target_class=None):
    """Get data loaders with advanced augmentation for better training."""
    
    # Define stronger data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
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
        shuffle=True,  # Enable shuffling
        num_workers=4,
    )
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    return train_dataloader, test_dataloader


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration for CIFAR-10
    config = create_config(
        dataset="cifar10",
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks
    )
    
    print(f"Creating dataloaders for CIFAR-10 with augmentation...")
    
    # Use the custom function with augmentation
    train_loader, val_loader = get_augmented_dataloaders(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        root_path=args.data_dir,
        train_subset_n=args.train_subset,
        test_subset_n=args.test_subset,
        target_class=args.target_class
    )
    
    # Print information about the dataset subsets
    if args.train_subset is not None:
        print(f"Using {args.train_subset} samples from the training set")
    if args.test_subset is not None:
        print(f"Using {args.test_subset} samples from the test set")
    if args.target_class is not None:
        print(f"Using only samples from class {args.target_class}")
    
    print("Initializing model...")
    # Create model with configuration
    model = TransformerDecoder(config)
    
    # For our final experiment, we'll use the successful learning rates from previous runs
    print(f"Using optimal learning rates: LR_WEIGHTS={INITIAL_LR_WEIGHTS}, LR_HIDDEN={INITIAL_LR_HIDDEN}")
    
    # Set up the optimizers
    optim_h = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(INITIAL_LR_HIDDEN, momentum=0.9)
        ))
    optim_w = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(INITIAL_LR_WEIGHTS, weight_decay=WEIGHT_DECAY, b1=0.9, b2=0.999)
        ), pxu.M(pxnn.LayerParam)(model))
    
    # Store validation losses to track progress
    val_losses = []
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
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
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Create a plot of the validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epoch + 2), val_losses, marker='o')
            plt.title('Validation Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.grid(True)
            os.makedirs("../results", exist_ok=True)
            plt.savefig(f"../results/validation_loss_curve_epoch_{epoch+1}.png")
            plt.close()
    
    print("Training complete!")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses))+1})")
    print(f"Total loss reduction: {(val_losses[0] - min(val_losses)) / val_losses[0] * 100:.2f}%")
    
    # Create a plot of the validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), val_losses, marker='o')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/final_validation_loss_curve.png")
    
    # Visualize results with different inference steps
    print("Visualizing reconstruction with different inference steps...")
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 4, 8], 
        use_corruption=False,
        num_images=4
    )
    
    # Visualize inpainting with different inference steps
    print("Visualizing inpainting with different inference steps...")
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