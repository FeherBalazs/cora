import jax
import jax.numpy as jnp
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
from typing import Callable, Optional, Dict, Any, Type
from jax.typing import DTypeLike
import jax.tree_util as jtu
from dataclasses import is_dataclass

from flax import nnx
from pcx.core._module import Module
from pcx.core._parameter import BaseParam
from pcx.core._static import StaticParam

from jflux.modules.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
)

# Try to import the TransformerConfig class if available
try:
    from src.decoder_transformer import TransformerConfig
    HAS_TRANSFORMER_CONFIG = True
except ImportError:
    HAS_TRANSFORMER_CONFIG = False
    TransformerConfig = type(None)  # Fallback placeholder


def is_px_static(obj):
    """Check if an object has been marked as static by px.static()"""
    # Objects marked with px.static typically have a special attribute or behavior
    return hasattr(obj, '__static_hash__') or hasattr(obj, '_static') or hasattr(obj, '_StaticGuard__val')


class PCXWrapper(Module):
    """
    Base wrapper for jflux components to make them compatible with PCX's parameter system.
    This wrapper converts flax nnx parameters to PCX LayerParam objects.
    
    This follows a similar approach to pcx.nn.Layer, adapted for Flax/nnx modules,
    with special handling for configuration objects and px.static objects.
    """
    
    def __init__(self, module, filter=lambda x: isinstance(x, jax.Array)):
        super().__init__()
        
        # Store the original module with parameters converted appropriately
        # Using an approach similar to Layer in pcx/nn/_layer.py but adapted for Flax
        self.module = jtu.tree_map(
            lambda w: pxnn.LayerParam(w) if filter(w) else StaticParam(w),
            module,
            is_leaf=lambda w: filter(w) or not hasattr(w, '__dict__') or 
                             is_px_static(w) or  # Handle px.static objects
                             (HAS_TRANSFORMER_CONFIG and isinstance(w, TransformerConfig)) or
                             is_dataclass(type(w))  # Handle all dataclasses
        )
        
    def __call__(self, *args, **kwargs):
        # Convert all parameter objects back to their values for the call
        # Following the same approach as in Layer.__call__
        module_with_arrays = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.module,
            is_leaf=lambda w: isinstance(w, BaseParam) or 
                             is_px_static(w) or  # Handle px.static objects
                             (HAS_TRANSFORMER_CONFIG and isinstance(w, TransformerConfig)) or
                             is_dataclass(type(w))  # Handle all dataclasses
        )
        
        # Call the module with the arrays
        return module_with_arrays(*args, **kwargs)


class PCXDoubleStreamBlock(PCXWrapper):
    """
    PCX-compatible wrapper for the jflux DoubleStreamBlock.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Create the original jflux DoubleStreamBlock
        original_block = DoubleStreamBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype
        )
        
        # Initialize the wrapper with the original block
        super().__init__(original_block)
        
        # Store configuration for reference
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
    def __call__(self, img: jax.Array, txt: jax.Array, vec: jax.Array, pe: jax.Array):
        return super().__call__(img, txt, vec, pe)


class PCXSingleStreamBlock(PCXWrapper):
    """
    PCX-compatible wrapper for the jflux SingleStreamBlock.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
        rngs: Optional[nnx.Rngs] = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Create the original jflux SingleStreamBlock
        original_block = SingleStreamBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=qk_scale,
            rngs=rngs,
            param_dtype=param_dtype
        )
        
        # Initialize the wrapper with the original block
        super().__init__(original_block)
        
        # Store configuration for reference
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
    def __call__(self, x: jax.Array, vec: jax.Array, pe: jax.Array):
        return super().__call__(x, vec, pe)


class PCXEmbedND(PCXWrapper):
    """
    PCX-compatible wrapper for the jflux EmbedND.
    """
    
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        # Create the original jflux EmbedND
        original_embed = EmbedND(dim=dim, theta=theta, axes_dim=axes_dim)
        
        # Initialize the wrapper with the original embed
        super().__init__(original_embed)
        
        # Store configuration for reference
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        
    def __call__(self, ids: jax.Array) -> jax.Array:
        return super().__call__(ids)


class PCXMLPEmbedder(PCXWrapper):
    """
    PCX-compatible wrapper for the jflux MLPEmbedder.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: Optional[nnx.Rngs] = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Create the original jflux MLPEmbedder
        original_embedder = MLPEmbedder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype
        )
        
        # Initialize the wrapper with the original embedder
        super().__init__(original_embedder)
        
        # Store configuration for reference
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)


class PCXLastLayer(PCXWrapper):
    """
    PCX-compatible wrapper for the jflux LastLayer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        rngs: Optional[nnx.Rngs] = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Create the original jflux LastLayer
        original_layer = LastLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            rngs=rngs,
            param_dtype=param_dtype
        )
        
        # Initialize the wrapper with the original layer
        super().__init__(original_layer)
        
        # Store configuration for reference
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
    def __call__(self, x: jax.Array, vec: jax.Array) -> jax.Array:
        return super().__call__(x, vec) 