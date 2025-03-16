import jax
import jax.numpy as jnp
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
from typing import Callable, Optional, Dict, Any
from jax.typing import DTypeLike
import jax.tree_util as jtu
from pcx.core import Module

from flax import nnx
from jflux.modules.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
)

class PCXWrapper(Module):
    """
    Base wrapper for jflux components to make them compatible with PCX's parameter system.
    This wrapper converts flax nnx parameters to PCX LayerParam objects.
    """
    
    def __init__(self, module):
        super().__init__()
        # Convert all jax arrays in the module to LayerParam objects
        # Use different tree handling based on JAX version
        if hasattr(jax, 'tree'):
            # JAX 0.5+ - Handle None values properly 
            self.module = px.static(jax.tree.map(
                lambda w: pxnn.LayerParam(w) if isinstance(w, jax.Array) else w,
                module,
                is_leaf=lambda w: w is None or isinstance(w, jax.Array)
            ))
        else:
            # JAX 0.4.x
            self.module = px.static(jtu.tree_map(
                lambda w: pxnn.LayerParam(w) if isinstance(w, jax.Array) else w,
                module,
                is_leaf=lambda w: isinstance(w, jax.Array)
            ))
        
    def __call__(self, *args, **kwargs):
        # Convert all LayerParam objects back to jax arrays for the call
        # Use different tree handling based on JAX version
        if hasattr(jax, 'tree'):
            # JAX 0.5+ - Handle None values properly
            module_with_arrays = jax.tree.map(
                lambda w: w.get() if isinstance(w, pxnn.LayerParam) else w,
                self.module,
                is_leaf=lambda w: w is None or isinstance(w, pxnn.LayerParam)
            )
        else:
            # JAX 0.4.x
            module_with_arrays = jtu.tree_map(
                lambda w: w.get() if isinstance(w, pxnn.LayerParam) else w,
                self.module,
                is_leaf=lambda w: isinstance(w, pxnn.LayerParam)
            )
        
        # Call the module with the arrays
        result = module_with_arrays(*args, **kwargs)
        
        return result


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