__all__ = ["StateParam", "StatefulLayer", "BatchNorm", "BatchNormPC"]

from typing import Hashable, Sequence

import jax
import jax.tree_util as jtu
import equinox as eqx
import jax.numpy as jnp
import jax.lax as lax

from ..core._module import Module
from ..core._parameter import BaseParam, Param
from ..core._static import StaticParam
from ._parameter import LayerParam


class StateParam(Param):
    pass


class StatefulLayer(Module):
    def __init__(self, cls, *args, filter=eqx._filters.is_array, **kwargs) -> None:
        super().__init__()

        self.nn, self.state = eqx.nn.make_with_state(cls)(*args, **kwargs)

        self.nn = jtu.tree_map(
            lambda w: LayerParam(w) if filter(w) else StaticParam(w),
            self.nn,
        )

        # print(self.nn)
        # print(self.state)

        # We opt for a single StateParam to encapsule all the state data. This limits the
        # masking that can be done on it (as you either select all or nothing). For a for
        # fine-grained selection, a per-array StateParam approach must be used.
        #
        # self.state = jtu.tree_map(
        #     lambda w: StateParam(w) if filter(w) else StaticParam(w),
        #     self.state,
        # )
        #
        self.state = StateParam(self.state)

    def __call__(self, *args, key=None, **kwargs):
        _nn = jtu.tree_map(
            lambda w: w.get() if isinstance(w, BaseParam) else w,
            self.nn,
            is_leaf=lambda w: isinstance(w, BaseParam),
        )

        _r, _state = _nn(*args, self.state.get(), **kwargs, key=key)

        # Alternative per-array StateParam approach
        #
        # jtu.tree_map(
        #     lambda p, v: p.set(v) if isinstance(p, StateParam) else None,
        #     self.state,
        #     _state,
        #     is_leaf=lambda w: isinstance(w, BaseParam),
        # )
        #
        self.state.set(_state)

        return _r


class BatchNorm(StatefulLayer):
    def __init__(
        self,
        input_size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-05,
        channelwise_affine: bool = True,
        momentum: float = 0.1,
        inference: bool = False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            eqx.nn.BatchNorm,
            input_size,
            axis_name,
            eps,
            channelwise_affine,
            momentum,
            inference,
            dtype,
            **kwargs,
        )


class BatchNormPC(Module):
    """A custom BatchNorm layer that computes statistics over the batch axis (0)
    and does not rely on JAX's collective operations (e.g. pmean), avoiding
    the need for a named axis from `vmap`.
    """

    weight: LayerParam
    bias: LayerParam
    running_mean: StateParam
    running_var: StateParam

    momentum: StaticParam
    eps: StaticParam
    
    inference: StateParam

    def __init__(
        self,
        input_size: int,
        axis_name: Hashable | Sequence[Hashable] = None, # Kept for API compatibility but ignored.
        eps: float = 1e-05,
        channelwise_affine: bool = True, # Kept for API compatibility. Assumed True.
        momentum: float = 0.1,
        inference: bool = False,
        dtype=None, # Kept for API compatibility.
        **kwargs,
    ):
        super().__init__()
        if not channelwise_affine:
            # This implementation only supports affine transformation.
            raise NotImplementedError("BatchNorm without channelwise_affine is not supported.")

        self.weight = LayerParam(jnp.ones(input_size, dtype=dtype))
        self.bias = LayerParam(jnp.zeros(input_size, dtype=dtype))
        self.running_mean = StateParam(jnp.zeros(input_size, dtype=dtype))
        self.running_var = StateParam(jnp.ones(input_size, dtype=dtype))

        self.momentum = StaticParam(momentum)
        self.eps = StaticParam(eps)
        self.inference = StateParam(jnp.array(inference))

    def __call__(self, x: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """
        Assumes input `x` has shape (batch, features).
        Calculates statistics along axis 0.
        Uses jax.lax.cond to be JIT-compatible.
        """

        def _inference_path(operand):
            # Use running averages for normalization.
            mean = self.running_mean.get()
            var = self.running_var.get()
            # Normalize
            x_norm = (operand - mean) / jnp.sqrt(var + self.eps.get())
            # Scale and shift
            output = x_norm * self.weight.get() + self.bias.get()
            # In inference mode, we don't update the state.
            current_state = {"running_mean": mean, "running_var": var, "inference": self.inference.get()}
            return output, current_state

        def _training_path(operand):
            # Use batch statistics for normalization and update running averages.
            batch_mean = jnp.mean(operand, axis=0)
            batch_var = jnp.var(operand, axis=0)
            
            # Update running statistics
            running_mean = self.running_mean.get()
            running_var = self.running_var.get()
            momentum = self.momentum.get()
            
            new_running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            new_running_var = (1 - momentum) * running_var + momentum * batch_var
            
            # Normalize
            x_norm = (operand - batch_mean) / jnp.sqrt(batch_var + self.eps.get())
            # Scale and shift
            output = x_norm * self.weight.get() + self.bias.get()

            new_state = {"running_mean": new_running_mean, "running_var": new_running_var, "inference": self.inference.get()}
            return output, new_state

        # The predicate `self.inference.get()` is a JAX array, so we use `lax.cond`.
        output, new_state = jax.lax.cond(
            self.inference.get(),
            _inference_path,
            _training_path,
            x
        )
        
        return output, new_state
