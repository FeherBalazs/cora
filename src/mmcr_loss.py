import jax
import jax.numpy as jnp


def calculate_mmcr_loss_for_vode(z: jax.Array, b_orig: int, n_views: int, mmcr_lambda: float):
    """Calculates the MMCR loss for a set of representations 'z'.
    
    Args:
        z: Input representations of shape (b_orig * n_views, projector_dim).
        b_orig: Original batch size.
        n_views: Number of views/augmentations per sample.
        mmcr_lambda: Weight for the local nuclear norm term.
    
    Returns:
        The MMCR loss value.
    """
    # 1. Normalize representations onto the unit sphere
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
    
    # 2. Reshape and calculate centroids
    z_grouped = z.reshape((b_orig, n_views, -1))  # (b_orig, n_views, projector_dim)
    c = z_grouped.mean(axis=1)  # Centroids, shape (b_orig, projector_dim)
    
    # 3. Calculate singular values for centroids
    s_c = jnp.linalg.svd(c, compute_uv=False)
    
    # 4. Conditionally calculate local nuclear norm
    if mmcr_lambda != 0.0:
        # Calculate singular values for each object's representation matrix
        s_z_b = jax.vmap(lambda x: jnp.linalg.svd(x, compute_uv=False))(z_grouped)
        # s_z_b has shape (b_orig, min(n_views, projector_dim))
        local_nuc = jnp.sum(s_z_b)
    else:
        local_nuc = 0.0
    
    # 5. Calculate loss according to MMCR (minimizing this value)
    loss = -jnp.sum(s_c) + (mmcr_lambda / b_orig) * local_nuc
    
    return loss