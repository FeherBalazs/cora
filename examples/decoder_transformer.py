import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import jax
import jax.numpy as jnp
import optax
import pcx.utils as pxu

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import the TransformerDecoder and utility functions
from decoder_transformer import (
    TransformerDecoder, 
    TransformerConfig,
    create_config_by_dataset,
    train, 
    eval, 
    eval_on_batch_partial,
    visualize_reconstruction
)
# Import PCX-compatible transformer components (though not directly used in this example)
from pcx_transformer import PCXDoubleStreamBlock, PCXEmbedND

# Set up basic parameters
BATCH_SIZE = 16
LATENT_DIM = 512
NUM_EPOCHS = 10
INFERENCE_STEPS = 24  # Number of inference steps for each batch
LR_WEIGHTS = 1e-4
LR_HIDDEN = 1e-2

def generate_synthetic_data(image_shape, num_samples=1000):
    """Generate synthetic data for testing the model"""
    # Create simple shapes (circles, squares, triangles) of different colors
    data = np.zeros((num_samples, *image_shape), dtype=np.float32)
    
    for i in range(num_samples):
        # Choose a random background color (slightly noisy)
        bg_color = np.random.uniform(0.0, 0.1, size=image_shape[0])
        
        # Fill the image with the background color
        for c in range(image_shape[0]):
            data[i, c] = bg_color[c]
        
        # Choose a shape type (0: circle, 1: square, 2: triangle)
        shape_type = np.random.randint(0, 3)
        
        # Choose a foreground color
        fg_color = np.random.uniform(0.5, 1.0, size=image_shape[0])
        
        # Choose position and size
        center_x = np.random.randint(8, image_shape[2]-8)
        center_y = np.random.randint(8, image_shape[1]-8)
        size = np.random.randint(6, 12)
        
        # Create shape
        if shape_type == 0:  # Circle
            for x in range(max(0, center_x - size), min(image_shape[2], center_x + size)):
                for y in range(max(0, center_y - size), min(image_shape[1], center_y + size)):
                    if (x - center_x)**2 + (y - center_y)**2 <= size**2:
                        for c in range(image_shape[0]):
                            data[i, c, y, x] = fg_color[c]
        
        elif shape_type == 1:  # Square
            x1, y1 = max(0, center_x - size//2), max(0, center_y - size//2)
            x2, y2 = min(image_shape[2], center_x + size//2), min(image_shape[1], center_y + size//2)
            for c in range(image_shape[0]):
                data[i, c, y1:y2, x1:x2] = fg_color[c]
        
        else:  # Triangle
            height = size
            half_base = size // 2
            for x in range(max(0, center_x - half_base), min(image_shape[2], center_x + half_base)):
                # Calculate max height at this x position
                rel_x = abs(x - center_x)
                max_y = int(height * (1 - rel_x / half_base)) if half_base > 0 else 0
                for y in range(max(0, center_y - max_y), min(image_shape[1], center_y)):
                    for c in range(image_shape[0]):
                        data[i, c, y, x] = fg_color[c]
    
    return data

def create_dataloaders(config, batch_size):
    """Create training and validation dataloaders"""
    # Generate synthetic data
    print("Generating synthetic data...")
    all_data = generate_synthetic_data(image_shape=config.image_shape, num_samples=1200)
    
    # Split into train and validation
    train_data = all_data[:1000]
    val_data = all_data[1000:]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_data))
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_data))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def visualize_results(model, optim_h, dataloader, num_samples=4):
    """Visualize original and reconstructed images"""
    # Get some random samples from the dataloader
    images, _ = next(iter(dataloader))
    images = images[:num_samples].numpy()
    
    # Run inference
    _, reconstructed = eval_on_batch_partial(
        use_corruption=False, 
        corrupt_ratio=0.5, 
        T=INFERENCE_STEPS, 
        x=images, 
        model=model, 
        optim_h=optim_h
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i in range(num_samples):
        # Plot original
        img_orig = np.transpose(images[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0.0, 1.0)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Plot reconstruction
        img_recon = np.transpose(reconstructed[i], (1, 2, 0))
        img_recon = np.clip(img_recon, 0.0, 1.0)
        axes[1, i].imshow(img_recon)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.show()

def visualize_inpainting(model, optim_h, dataloader, corrupt_ratio=0.5, num_samples=4):
    """Visualize image inpainting results"""
    # Get some random samples from the dataloader
    images, _ = next(iter(dataloader))
    images = images[:num_samples].numpy()
    
    # Run inference with corruption (inpainting)
    _, reconstructed = eval_on_batch_partial(
        use_corruption=True, 
        corrupt_ratio=corrupt_ratio, 
        T=INFERENCE_STEPS, 
        x=images, 
        model=model, 
        optim_h=optim_h
    )
    
    # Create corrupted images for visualization
    corrupted = images.copy()
    corrupt_height = int(corrupt_ratio * model.config.image_shape[1])
    for i in range(num_samples):
        corrupted[i, :, corrupt_height:, :] = 0.0
    
    # Visualize results
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    
    for i in range(num_samples):
        # Plot original
        img_orig = np.transpose(images[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0.0, 1.0)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Plot corrupted
        img_corr = np.transpose(corrupted[i], (1, 2, 0))
        img_corr = np.clip(img_corr, 0.0, 1.0)
        axes[1, i].imshow(img_corr)
        axes[1, i].set_title("Corrupted")
        axes[1, i].axis("off")
        
        # Plot reconstruction
        img_recon = np.transpose(reconstructed[i], (1, 2, 0))
        img_recon = np.clip(img_recon, 0.0, 1.0)
        axes[2, i].set_title("Inpainted")
        axes[2, i].imshow(img_recon)
        axes[2, i].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # Select dataset to use
    dataset_name = "cifar10"  # Options: "fashionmnist", "cifar10", "imagenet"
    
    # Create configuration based on dataset
    config = create_config_by_dataset(
        dataset_name=dataset_name,
        latent_dim=LATENT_DIM,
        num_blocks=6
    )
    
    print(f"Creating dataloaders for {dataset_name}...")
    train_loader, val_loader = create_dataloaders(config=config, batch_size=BATCH_SIZE)
    
    print("Initializing model...")
    # Create model with configuration
    model = TransformerDecoder(config)
    
    # Create optimizers
    optim_w = pxu.Optim(
        core_optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=LR_WEIGHTS, weight_decay=1e-4)
        )
    )
    
    optim_h = pxu.Optim(
        core_optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=LR_HIDDEN)
        )
    )
    
    # Train model
    print(f"Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train for one epoch
        train(train_loader, INFERENCE_STEPS, model=model, optim_w=optim_w, optim_h=optim_h)
        
        # Evaluate on validation set
        val_loss = eval(val_loader, INFERENCE_STEPS, model=model, optim_h=optim_h)
        print(f"Validation loss: {val_loss:.6f}")
    
    print("Training complete!")
    
    # Visualize results
    print("Visualizing reconstruction results...")
    visualize_results(model, optim_h, val_loader)
    
    print("Visualizing inpainting results...")
    visualize_inpainting(model, optim_h, val_loader)
    
    # Visualize reconstruction with different inference steps
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
