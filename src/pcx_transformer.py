import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Any

# PCX imports
import pcx
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu


from pcx.core._module import Module
from pcx.core._random import RandomKeyGenerator, RKG
from pcx.core._parameter import BaseParam
from pcx.core._static import StaticParam
from pcx.nn._parameter import LayerParam

from flax import nnx

# jflux component imports
from jflux.modules.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    SingleStreamBlockStandard,
    EmbedND,
    LastLayer,
    LastLayerStandard,
    MLPEmbedder,
)


########################################################################################################################
#
# FLAX LAYER
#
# pcx flax layers are a thin wrapper around Flax modules that initializes parameters with a sample input,
# replaces JAX arrays with LayerParam instances, and provides a __call__ method to unwrap and apply the module.
#
########################################################################################################################


class FlaxLayer(Module):
    """A generic wrapper for Flax modules, integrating them into the PCX framework."""

    def __init__(
        self,
        cls,
        *args,
        sample_input: Any,
        rngs: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        """
        Initialize the Flax module and wrap its parameters.

        Args:
            cls: The Flax module class to wrap (e.g., SingleStreamBlock).
            *args: Positional arguments for the Flax module constructor.
            sample_input: A sample input tensor to initialize parameter shapes.
            rngs: Random key generator for initialization.
            **kwargs: Keyword arguments for the Flax module constructor.
        """
        super().__init__()
        # Create the flax module
        self.flax_module = self._create_flax_module(cls, *args, **kwargs)
        
        # Initialize parameters with a random key and sample input
        self.params = self.flax_module.init(rngs(), sample_input)["params"]
        # Wrap parameters in LayerParam for PCX compatibility
        self.params = jtu.tree_map(lambda w: LayerParam(w), self.params)

    def _create_flax_module(self, cls, *args, **kwargs):
        """Create the flax module. Can be overridden by subclasses."""
        return cls(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute the forward pass by unwrapping parameters and calling apply."""
        # Unwrap parameters from LayerParam instances
        unwrapped_params = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.params,
            is_leaf=lambda w: isinstance(w, BaseParam),
        )
        # Call the Flax module's apply method
        return self.flax_module.apply({"params": unwrapped_params}, *args, **kwargs)


class PCXSingleStreamBlockStandard(FlaxLayer):
    """A PCX wrapper for the Flax SingleStreamBlock module."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        sample_input: Any,
        rngs: RandomKeyGenerator = RKG,
        param_dtype: Any = None,
    ):
        """
        Initialize a SingleStreamBlock with PCX parameter wrapping.

        Args:
            hidden_size: Size of the hidden state.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for MLP hidden size.
            sample_input: A sample input tensor to initialize parameter shapes.
            rngs: Random key generator for initialization.
            param_dtype: Data type for parameters (e.g., jnp.float32).
        """
        # Convert RandomKeyGenerator to nnx.Rngs for SingleStreamBlockStandard
        nnx_rngs = nnx.Rngs(jax.random.PRNGKey(0) if rngs is None else rngs())
        
        super().__init__(
            cls=SingleStreamBlockStandard,
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            sample_input=sample_input,
            rngs=rngs,  # Original rngs for FlaxLayer initialization
            param_dtype=param_dtype if param_dtype is not None else jnp.float32,
            # Pass nnx_rngs directly to the SingleStreamBlockStandard
            _nnx_rngs=nnx_rngs,
        )
        
    def _create_flax_module(self, cls, *args, **kwargs):
        """Create the flax module with the correct rngs format."""
        # Extract nnx_rngs from kwargs
        nnx_rngs = kwargs.pop('_nnx_rngs', None)
        
        # Update kwargs to use nnx_rngs as 'rngs' parameter
        if nnx_rngs is not None:
            kwargs['rngs'] = nnx_rngs
            
        return cls(*args, **kwargs)

