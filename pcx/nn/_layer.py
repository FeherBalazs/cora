__all__ = [
    "Layer",
    "Linear",
    "Conv",
    "Conv2d",
    "ConvTranspose",
    "Pool",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptivePool",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "Dropout",
    "LayerNorm",
    "PatchEmbedding",
    "MultiHeadAttention",
    "TransformerBlock"
]


from typing import Tuple, Sequence, Callable, Optional, List

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import einops
import pcx as px

from ..core._module import Module
from ..core._random import RandomKeyGenerator, RKG
from ..core._parameter import BaseParam
from ..core._static import StaticParam
from ._parameter import LayerParam


########################################################################################################################
#
# LAYER
#
# pcax layers are a thin wrapper around equinox layers that replaces all jax.Arrays with LayerParam instances.
# In this file only stateless layers are implemented as they don't need any particular ad-hoc adaptation.
########################################################################################################################


# Core #################################################################################################################


class Layer(Module):
    def __init__(
        self,
        cls,
        *args,
        filter=eqx._filters.is_array,
        **kwargs,
    ):
        super().__init__()
        self.nn = jtu.tree_map(
            lambda w: LayerParam(w) if filter(w) else StaticParam(w),
            cls(*args, **kwargs),
        )

    def __call__(self, *args, key=None, **kwargs):
        # Can do this, since nn is stateless
        _nn = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.nn,
            is_leaf=lambda w: isinstance(w, BaseParam),
        )

        return _nn(*args, **kwargs, key=key)


# Common Layers ########################################################################################################


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(eqx.nn.Linear, in_features, out_features, bias, key=rkg(), **kwargs)


class Conv(Layer):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.Conv,
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=rkg(),
            **kwargs,
        )


class Conv2d(Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            rkg=rkg,
            **kwargs,
        )


class ConvTranspose(Layer):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.ConvTranspose,
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=rkg(),
            **kwargs,
        )


# Pooling ##############################################################################################################


class Pool(Layer):
    def __init__(
        self,
        init: int | float | jax.Array,
        operation: Callable[[jax.Array, jax.Array], jax.Array],
        num_spatial_dims: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        use_ceil: bool = False,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.Pool,
            init,
            operation,
            num_spatial_dims,
            kernel_size,
            stride,
            padding,
            use_ceil,
            **kwargs,
        )


class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        use_ceil: bool = False,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.MaxPool2d, kernel_size, stride, padding, use_ceil, **kwargs
        )


class AvgPool2d(Layer):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[Tuple[int, int]] = 0,
        use_ceil: bool = False,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.AvgPool2d, kernel_size, stride, padding, use_ceil, **kwargs
        )


class AdaptivePool(Layer):
    def __init__(
        self,
        target_shape: int | Sequence[int],
        num_spatial_dims: int,
        operation: Callable,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.AdaptivePool, target_shape, num_spatial_dims, operation, **kwargs
        )


class AdaptiveAvgPool2d(Layer):
    def __init__(
        self,
        target_shape: int | Sequence[int],
        **kwargs,
    ):
        super().__init__(eqx.nn.AdaptiveAvgPool2d, target_shape, **kwargs)


class AdaptiveMaxPool2d(Layer):
    def __init__(
        self,
        target_shape: int | Sequence[int],
        **kwargs,
    ):
        super().__init__(eqx.nn.AdaptiveMaxPool2d, target_shape, **kwargs)


# Dropout ##############################################################################################################


class Dropout(Layer):
    def __init__(self, p: float = 0.5, inference: bool = False, **kwargs):
        super().__init__(eqx.nn.Dropout, p, inference, **kwargs)


# Normalisation ########################################################################################################


class LayerNorm(Layer):
    def __init__(
        self,
        shape: Tuple[int, ...] | None = None,
        eps: float = 1e-05,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=None,
        *,
        elementwise_affine: bool = True,
    ):
        super().__init__(
            eqx.nn.LayerNorm,
            shape,
            eps,
            use_weight,
            use_bias,
            dtype,
            elementwise_affine=elementwise_affine,
        )


# Transformer Components ##################################################################################################

class PatchEmbedding(Layer):
    """
    Layer that converts images into patch embeddings for Vision Transformers.
    """
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_size: int,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.Linear,
            patch_size * patch_size * input_channels,
            embed_dim,
            key=rkg(),
            **kwargs,
        )
        # Store parameters needed for patching operation
        self.patch_size = px.static(patch_size)
        self.input_channels = px.static(input_channels)
        self.embed_dim = px.static(embed_dim)

    def __call__(self, x, key=None):
        # x shape: (C, H, W)
        # Rearrange image into patches and flatten each patch
        x = einops.rearrange(
            x,
            "c (h p1) (w p2) -> (h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        # Apply projection to each flattened patch
        x = super().__call__(x, key=key)
        return x


class MultiHeadAttention(Layer):
    """
    Multi-head attention layer for transformer architectures.
    Wrapper around Equinox's MultiheadAttention.
    """
    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: int | None = None,
        value_size: int | None = None,
        output_size: int | None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        dtype = None,
        *,
        rkg: RandomKeyGenerator = RKG,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.MultiheadAttention,
            num_heads,
            query_size,
            key_size,
            value_size,
            output_size,
            qk_size,
            vo_size,
            use_query_bias,
            use_key_bias,
            use_value_bias,
            use_output_bias,
            dropout_p,
            inference,
            dtype,
            key=rkg(),
            **kwargs,
        )


class TransformerBlock(Module):
    """
    Transformer block with self-attention and feedforward network.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.layer_norm1 = LayerNorm(input_shape)
        self.layer_norm2 = LayerNorm(input_shape)
        self.attention = MultiHeadAttention(num_heads, input_shape)

        self.linear1 = Linear(input_shape, hidden_dim)
        self.linear2 = Linear(hidden_dim, input_shape)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def __call__(self, x, enable_dropout: bool=False):
        
        # TODO: Fix layer norm
        # Apply layer normalization
        # input_x = jax.vmap(self.layer_norm1)(x)
        input_x = x

        # Apply self-attention and add residual connection
        x = x + self.attention(input_x, input_x, input_x)

        # Apply layer normalization and feedforward network
        # input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        input_x = self.dropout1(input_x, inference=not enable_dropout)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout)

        x = x + input_x

        return x
    