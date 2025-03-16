# Running Cora on GPU with JAX 0.5+

This guide explains how to run Cora on a GPU server using the updated JAX 0.5+ support.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit installed
- Git repository access

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_REPO/cora.git
   cd cora
   ```

2. Build and run the Docker container:
   ```bash
   cd docker
   chmod +x run.sh
   ./run.sh
   ```

3. Test GPU support:
   ```bash
   # From the docker directory
   ./run.sh test
   ```

## Docker Environment

The Docker environment includes:
- Ubuntu 22.04
- CUDA 12.2 with cuDNN 8
- Python 3.11
- JAX 0.5.2 with GPU support
- Equinox 0.11.12
- Optax 0.2.4
- Flax 0.10.4

## Running Experiments

Once inside the Docker container, you can run training scripts:

```bash
# Run the transformer example
python examples/run_transformer.py --dataset fashionmnist --batch-size 16 --num-blocks 4 --epochs 10
```

## Troubleshooting

### Verifying GPU Access

Inside the Docker container, you can verify that JAX can see the GPU:

```python
import jax
print(jax.devices())  # Should list GPU devices
```

### Common Issues

1. **Memory Errors**: If you encounter CUDA out-of-memory errors, try reducing batch sizes or model sizes.

2. **Missing GPU**: If JAX can't see the GPU, verify that:
   - NVIDIA drivers are installed correctly
   - The Docker container has GPU access
   - The CUDA version in the Dockerfile matches your driver compatibility

3. **Version Conflicts**: If you see JAX/CUDA version conflicts, check that the JAX and CUDA versions are compatible.

### Performance Tips

- Set `XLA_PYTHON_CLIENT_PREALLOCATE="false"` to prevent JAX from allocating 90% of GPU memory upfront (this is already done in the Docker environment)
- For multi-GPU setups, you can use JAX's distributed options
- Use `jax.device_get()` or `.block_until_ready()` to force the completion of computations when timing operations

## Running with Custom Data

To use your own data with Cora, mount additional volumes when running the Docker container:

```bash
docker run --gpus all -it \
  -v /path/to/cora:/home/cora/workspace \
  -v /path/to/data:/home/cora/data \
  cora:latest /bin/bash
```

Then access the data from `/home/cora/data` in your scripts. 