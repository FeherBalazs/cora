import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional


def get_sinusoidal_1d(positions, dim, theta=10000.0):
    """
    Generate sinusoidal positional encoding for a 1D sequence.
    
    Args:
        positions: 1D array of positions
        dim: Dimension of the embedding (must be even)
        theta: Parameter for frequency scaling
        
    Returns:
        Encoding with shape (len(positions), dim)
    """
    # Make sure dim is even for sine/cosine pairs
    dim = dim - dim % 2
    
    # Create position encodings 
    position = positions[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(theta) / dim))
    
    # Apply sin to even indices and cos to odd indices
    pe = jnp.zeros((positions.shape[0], dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pe


def create_2d_positional_encoding(
    height: int, 
    width: int, 
    hidden_size: int, 
    patch_size: int, 
    theta: float = 10000.0
) -> jnp.ndarray:
    """
    Creates 2D sinusoidal positional encodings for image transformer models.
    
    Args:
        height: Image height
        width: Image width
        hidden_size: Hidden dimension size of the transformer
        patch_size: Size of patches
        theta: Parameter for frequency scaling
        
    Returns:
        Positional encodings with shape (num_patches, hidden_size)
    """
    # Calculate number of patches in each dimension
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    # Create position indices for both dimensions
    h_pos = jnp.arange(0, h_patches)
    w_pos = jnp.arange(0, w_patches)
    
    # Create 2D position grid
    pos_h, pos_w = jnp.meshgrid(h_pos, w_pos, indexing='ij')
    
    # Flatten to match transformer input sequence
    h_flat = pos_h.reshape(-1)
    w_flat = pos_w.reshape(-1)
    
    # Get sinusoidal encoding for each dimension
    h_enc = get_sinusoidal_1d(h_flat, dim=hidden_size // 2, theta=theta)
    w_enc = get_sinusoidal_1d(w_flat, dim=hidden_size // 2, theta=theta)
    
    # Combine encodings
    pos_enc = jnp.concatenate([h_enc, w_enc], axis=1)
    
    # Ensure the positional encoding has the correct size
    if pos_enc.shape[1] < hidden_size:
        # Pad if needed
        padding = hidden_size - pos_enc.shape[1]
        pos_enc = jnp.pad(pos_enc, ((0, 0), (0, padding)))
    elif pos_enc.shape[1] > hidden_size:
        # Truncate if needed
        pos_enc = pos_enc[:, :hidden_size]
        
    return pos_enc


def create_3d_positional_encoding(
    frames: int,
    height: int, 
    width: int, 
    hidden_size: int, 
    patch_size: int, 
    theta: float = 10000.0
) -> jnp.ndarray:
    """
    Creates 3D sinusoidal positional encodings for video transformer models.
    
    Args:
        frames: Number of frames in video
        height: Image height
        width: Image width
        hidden_size: Hidden dimension size of the transformer
        patch_size: Size of patches
        theta: Parameter for frequency scaling
        
    Returns:
        Positional encodings with shape (num_patches, hidden_size)
    """
    # Calculate number of patches in each dimension
    frame_patches = frames
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    # Create position indices for all three dimensions
    t_pos = jnp.arange(0, frame_patches)
    h_pos = jnp.arange(0, h_patches)
    w_pos = jnp.arange(0, w_patches)
    
    # Create 3D position grid
    pos_t, pos_h, pos_w = jnp.meshgrid(t_pos, h_pos, w_pos, indexing='ij')
    
    # Flatten to match transformer input sequence
    t_flat = pos_t.reshape(-1)
    h_flat = pos_h.reshape(-1)
    w_flat = pos_w.reshape(-1)
    
    # Get sinusoidal encoding for each dimension
    t_enc = get_sinusoidal_1d(t_flat, dim=hidden_size // 3, theta=theta)
    h_enc = get_sinusoidal_1d(h_flat, dim=hidden_size // 3, theta=theta)
    w_enc = get_sinusoidal_1d(w_flat, dim=hidden_size // 3, theta=theta)
    
    # Combine encodings
    pos_enc = jnp.concatenate([t_enc, h_enc, w_enc], axis=1)
    
    # Ensure the positional encoding has the correct size
    if pos_enc.shape[1] < hidden_size:
        # Pad if needed
        padding = hidden_size - pos_enc.shape[1]
        pos_enc = jnp.pad(pos_enc, ((0, 0), (0, padding)))
    elif pos_enc.shape[1] > hidden_size:
        # Truncate if needed
        pos_enc = pos_enc[:, :hidden_size]
        
    return pos_enc


def create_positional_encoding(
    image_shape: Tuple, 
    hidden_size: int, 
    patch_size: int, 
    is_video: bool = False,
    num_frames: int = None,
    theta: float = 10000.0
) -> jnp.ndarray:
    """
    Creates sinusoidal positional encodings for transformer models.
    Handles both image (2D) and video (3D) inputs.
    
    Args:
        image_shape: Shape of the image/video (channels, height, width) or (frames, channels, height, width)
        hidden_size: Hidden dimension size of the transformer
        patch_size: Size of patches
        is_video: Whether the input is a video
        num_frames: Number of frames if video (can be inferred from image_shape if it has 4 dimensions)
        theta: Parameter for frequency scaling
        
    Returns:
        Positional encodings with shape (num_patches, hidden_size)
    """
    if is_video or len(image_shape) == 4:
        # Video case: extract dimensions
        if len(image_shape) == 4:
            frames, _, height, width = image_shape
        else:
            frames = num_frames
            _, height, width = image_shape
            
        return create_3d_positional_encoding(
            frames=frames,
            height=height,
            width=width,
            hidden_size=hidden_size,
            patch_size=patch_size,
            theta=theta
        )
    else:
        # Image case: extract dimensions
        _, height, width = image_shape
        
        return create_2d_positional_encoding(
            height=height,
            width=width,
            hidden_size=hidden_size,
            patch_size=patch_size,
            theta=theta
        ) 