import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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

# Set up basic parameters
BATCH_SIZE = 16
LATENT_DIM = 512
NUM_EPOCHS = 2
NUM_BLOCKS = 3
INFERENCE_STEPS = 1  # Number of inference steps for each batch
LR_WEIGHTS = 1e-4
LR_HIDDEN = 1e-2

def create_config(dataset="cifar10", latent_dim=512, num_blocks=3):
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
            use_noise=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def create_dataloaders(config, batch_size, use_real_data=True, data_dir='./data'):
    """Create training and validation dataloaders"""
    from torchvision import datasets, transforms
    
    if use_real_data and config.image_shape[0] == 3 and config.image_shape[1] == 32 and config.image_shape[2] == 32:
        print("Loading CIFAR-10 data...")
        # CIFAR-10 preprocessing - normalization to [0,1] range
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] and CHW format
        ])
        
        # Download and load CIFAR-10 training set
        train_dataset = datasets.CIFAR10(
            root=data_dir, 
            train=True,
            download=True, 
            transform=transform
        )
        
        # Download and load CIFAR-10 test set
        val_dataset = datasets.CIFAR10(
            root=data_dir, 
            train=False,
            download=True, 
            transform=transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    else:
        raise ValueError("Only CIFAR-10 is supported currently.")


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
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store datasets (default: ./data)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration for CIFAR-10
    config = create_config(
        dataset="cifar10",
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks
    )
    
    print(f"Creating dataloaders for CIFAR-10...")
    train_loader, val_loader = create_dataloaders(
        config=config, 
        batch_size=args.batch_size, 
        use_real_data=True,
        data_dir=args.data_dir
    )
    
    print("Initializing model...")
    # Create model with configuration
    model = TransformerDecoder(config)
    
    # Create optimizers
    optim_h = pxu.Optim(lambda: optax.sgd(5e-1, momentum=0.1))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train(train_loader, args.inference_steps, model=model, optim_w=optim_w, optim_h=optim_h)
        
        # Evaluate on validation set
        val_loss = eval(val_loader, args.inference_steps, model=model, optim_h=optim_h)
        print(f"Validation loss: {val_loss:.6f}")
    
    print("Training complete!")
    
    # Visualize results with different inference steps
    print("Visualizing reconstruction with different inference steps...")
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 4, 8, 16, 32], 
        use_corruption=False,
        num_images=4
    )
    
    # Visualize inpainting with different inference steps
    print("Visualizing inpainting with different inference steps...")
    visualize_reconstruction(
        model, 
        optim_h, 
        val_loader, 
        T_values=[1, 4, 8, 16, 32], 
        use_corruption=True,
        corrupt_ratio=0.5,
        num_images=4
    )

if __name__ == "__main__":
    main() 