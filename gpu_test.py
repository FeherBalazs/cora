#!/usr/bin/env python
"""
GPU Test Script for Cora

This script verifies that JAX is properly detecting and using the GPU.
It also runs a simple transformer test to ensure the model works on GPU.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from src.decoder_transformer import (
    TransformerDecoder,
    TransformerConfig,
    forward,
    eval_on_batch_partial
)
import pcx.utils as pxu
import optax

def print_device_info():
    """Print information about available devices."""
    print("\n====== JAX Device Information ======")
    print(f"JAX version: {jax.__version__}")
    print(f"Number of devices: {jax.device_count()}")
    print(f"Device list: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print("===================================\n")

def test_gpu_performance():
    """Test GPU performance with a simple matrix multiplication."""
    print("\n====== Testing GPU Performance ======")
    
    # Create large matrices
    n = 5000
    print(f"Creating {n}x{n} matrices...")
    x = jnp.ones((n, n))
    
    # Time matrix multiplication
    print("Running matrix multiplication...")
    start_time = time.time()
    result = jnp.dot(x, x)
    # Force execution of the computation
    result.block_until_ready()
    elapsed = time.time() - start_time
    
    print(f"Matrix multiplication completed in {elapsed:.4f} seconds.")
    if elapsed < 0.5:
        print("✓ GPU acceleration is working properly.")
    else:
        print("⚠ Performance seems slow. GPU might not be used efficiently.")
    print("===================================\n")
    
    return result

def test_transformer_model():
    """Test running a simple transformer model."""
    print("\n====== Testing Transformer Model ======")
    
    # Create a small transformer config
    config = TransformerConfig(
        latent_dim=256,
        image_shape=(3, 32, 32),
        num_blocks=2,
        hidden_size=256,
        num_heads=4
    )
    
    print("Initializing transformer model...")
    model = TransformerDecoder(config)
    
    # Create optimizers
    print("Creating optimizers...")
    optim_w = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
        )
    )
    
    optim_h = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-2)
        )
    )
    
    # Create random input data
    print("Creating random input data...")
    batch_size = 4
    x = jnp.ones((batch_size, *config.image_shape)) * 0.5
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    loss, x_hat = eval_on_batch_partial(
        use_corruption=False,
        corrupt_ratio=0.5,
        T=5,  # 5 inference steps
        x=x,
        model=model,
        optim_h=optim_h
    )
    
    # Force execution
    if x_hat is not None:
        x_hat.block_until_ready()
    elapsed = time.time() - start_time
    
    if x_hat is not None:
        print(f"✓ Transformer inference completed in {elapsed:.4f} seconds.")
        print(f"  Output shape: {x_hat.shape}")
    else:
        print("⚠ Transformer inference failed.")
    
    print("===================================\n")

if __name__ == "__main__":
    print("\nRunning Cora GPU Test Script")
    print("===========================")
    
    # Print device information
    print_device_info()
    
    # Run GPU performance test
    _ = test_gpu_performance()
    
    # Test transformer model
    test_transformer_model()
    
    print("\nTest completed.") 