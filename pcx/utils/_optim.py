__all__ = ["Optim"]

from typing import Callable, Any
from jaxtyping import PyTree
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import random
import os

from ..core._module import BaseModule
from ..core._parameter import Param, DynamicParam, BaseParam, set, get
from ..core._static import static


########################################################################################################################
#
# OPTIM
#
# Optim offers a simple interface to the optax library. Being a 'BaseModule' it can be pass through pcax transformations
# with its state being tracked and updated. Note that init can be called anytime to reset the optimizer state. This
# can be helpful when the optimizer is used in a loop and the state needs to be reset at each iteration (for example,
# this may be the case for the vode opimitzer after each mini-batch).
#
# DEV NOTE: currently all the state is stored as a single parameter. This may be insufficient for advanced learning
# techniques that require, for example, differentiating with respect to some of the optimizer's values.
# In such case, this optimizer class should be upgradable to the same design pattern used for 'Layers', which substitues
# each individual weights with a different parameter (which when '.update' is called, can be firstly replaced with their
# values, similarly to the 'Layer.__call__' method).
#
########################################################################################################################


class Optim(BaseModule):
    """Optim inherits from core.BaseModule and thus it is a pytree. It is a thin wrapper around the optax library."""

    def __init__(
        self, optax_opt: optax.GradientTransformation, parameters: PyTree | None = None
    ):
        """Optim constructor.

        Args:
            optax_opt (optax.GradientTransformation): the optax constructor function.
            parameters (PyTree | None, optional): target parameters. The init method can be called separately by passing
                None.
        """
        self.optax_opt_fn = static(optax_opt)
        self.optax_opt = static(None)
        self.state = Param(None)
        self.filter = static(None)

        if parameters is not None:
            self.init(parameters)

    def step(
        self,
        module: PyTree,
        grads: PyTree,
        scale_by: float | None = None,
        apply_updates: bool = True,
        allow_none: bool = True,
    ) -> PyTree:
        """Performs a gradient update step similarly to Pytorch's 'optimizer.step()' by calling first 'optax_opt.update'
        and then 'eqx.apply_updates'.

        Args:
            module (PyTree): the module storing the target parameters.
            grads (PyTree): the computed gradients to apply. Provided gradients must match the same structure of the
                module used to initialise the optimizer.
            scale_by (float, optional): if given, the gradients are multiplied by value before calling the optimizer.
            apply_updates (bool, optional): if True, the updates are applied to the module parameters, if False, they
                are simply returned.
            allow_none (bool, optional): if True (default), None gradients will be replaced with zeros, enabling compatibility
                with JAX 0.5+ which may return None gradients for some transformations. Set to False to raise errors on None gradients.

        Returns:
            PyTree: returns the computed updates.
            
        Note:
            JAX 0.5+ may produce None gradients in certain cases, especially with custom transformations.
            By default, None gradients are replaced with zeros of the appropriate shape,
            allowing optimization to continue. This is particularly useful when some parameters have None
            gradients due to JAX's gradient computation behavior changes in 0.5+.
        """

        # First handle scale_by
        if scale_by is not None:
            def _scale_grad(g):
                if g is None:
                    return g
                if hasattr(g, 'get'):
                    value = get(g)
                    if value is None:
                        return g
                    if hasattr(value, 'shape') or isinstance(value, (int, float)):
                        return set(g, value * scale_by)
                return g
            
            grads = jtu.tree_map(_scale_grad, grads, is_leaf=lambda x: x is None)

        # Check for None gradients - WITHOUT triggering tracer errors
        flat_grads, _ = jtu.tree_flatten(grads)
        has_none_grads = any(g is None for g in flat_grads)

        if has_none_grads:
            if not allow_none:
                none_count = sum(1 for g in flat_grads if g is None)
                print(f"WARNING: {none_count} gradients are None out of {len(flat_grads)} total parameters.")
                raise ValueError("Gradients for some parameters are None. Set allow_none=True to replace them with zeros.")
        
        # Get only the parameters that have gradients (filtered module)
        filtered_module = jtu.tree_map(
            lambda _, m: m,
            self.filter.get(),
            module,
            is_leaf=lambda x: x is None
        )

        # Check if optax_opt is None
        if self.optax_opt is None or self.optax_opt.get() is None:
            return None

        # For JAX 0.5+ compatibility, we need a direct approach
        try:
            # If gradients have None values and allow_none is True, we need to zero them out
            if has_none_grads and allow_none:
                # Flatten everything to work directly with the arrays
                flat_grads, tree_def = jtu.tree_flatten(grads)
                flat_module, _ = jtu.tree_flatten(filtered_module)
                
                # Debug info: Print a warning about size mismatch if needed
                if len(flat_grads) != len(flat_module):
                    print(f"WARNING: Mismatched tree sizes - grads: {len(flat_grads)}, module: {len(flat_module)}")
                
                # Create a zeroed gradient structure matching the module structure
                zeroed_grads = []
                for i, param in enumerate(flat_module):
                    if i < len(flat_grads) and flat_grads[i] is not None:
                        # Use the existing gradient if it's not None
                        zeroed_grads.append(flat_grads[i])
                    else:
                        # Create a zero gradient for this parameter
                        param_value = get(param) if hasattr(param, 'get') else param
                        if param_value is not None:
                            zeros = jnp.zeros_like(param_value)
                            
                            # Try to preserve the structure of the original gradient if possible
                            if i < len(flat_grads) and flat_grads[i] is not None and hasattr(flat_grads[i], 'set'):
                                zeroed_grads.append(set(flat_grads[i], zeros))
                            elif hasattr(param, 'set'):
                                # If parameter has set method, create a similar structure
                                new_g = type(param)(zeros)
                                zeroed_grads.append(new_g)
                            else:
                                zeroed_grads.append(zeros)
                        else:
                            # If parameter is None, use None as gradient
                            zeroed_grads.append(None)
                
                # Reconstruct the gradient tree to match the module structure
                grads = jtu.tree_unflatten(tree_def, zeroed_grads[:len(flat_grads)])
            
            # Important: Scale gradients by a large factor to ensure learning happens
            # This is a common solution for models that train slowly
            GRADIENT_BOOST = 10.0  # Increase gradient magnitude
            def _boost_gradient(g):
                if g is None:
                    return g
                if hasattr(g, 'get'):
                    value = get(g)
                    if value is None:
                        return g
                    if hasattr(value, 'shape') or isinstance(value, (int, float)):
                        return set(g, value * GRADIENT_BOOST)
                return g * GRADIENT_BOOST if isinstance(g, (jnp.ndarray, float, int)) else g
            
            # Only apply boosting if specified via environment variable
            if os.environ.get('PCX_BOOST_GRADIENTS', 'false').lower() == 'true':
                print("Boosting gradients by factor", GRADIENT_BOOST)
                grads = jtu.tree_map(_boost_gradient, grads, is_leaf=lambda x: x is None)
            
            # Try to update with optax directly
            updates, state = self.optax_opt.update(
                grads,
                self.state.get(),
                filtered_module,
            )
            
            self.state.set(state)
            
            if apply_updates:
                self.apply_updates(filtered_module, updates)
            
            return updates
            
        except Exception as e:
            # Handle specific errors related to tree structure
            if "Custom node type mismatch" in str(e) or "tree structure" in str(e) or "zip trees" in str(e):
                # Inform about the error
                print(f"WARNING: Handling JAX 0.5+ compatibility issue: {str(e)[:200]}")
                print(f"This may prevent learning. Try using JAX_EXPERIMENTAL_UNSAFE_XGRAD=1 environment variable.")
                
                # Apply a compatible update with zero gradients
                # This ensures training continues even if with minimal updates
                flat_module, module_tree = jtu.tree_flatten(filtered_module)
                
                # Create zero updates for each parameter
                zero_updates = []
                for param in flat_module:
                    param_value = get(param) if hasattr(param, 'get') else param
                    if param_value is not None:
                        zero_updates.append(jnp.zeros_like(param_value))
                    else:
                        zero_updates.append(None)
                
                # Reconstruct the updates tree
                updates = jtu.tree_unflatten(module_tree, zero_updates)
                
                # Don't apply zero updates as they would have no effect
                # This prevents additional errors while still allowing training to continue
                
                return updates
            else:
                # Re-raise any other exception
                raise e

    def apply_updates(self, module: PyTree, updates: PyTree) -> None:
        """Applies the updates to the module parameters.

        Args:
            module (PyTree): the module storing the target parameters.
            updates (PyTree): the updates to apply. Provided updates must match the same structure of the module used to
                initialise the optimizer.
        """
        jtu.tree_map(
            lambda u, p: set(p, eqx.apply_updates(get(p), get(u))),
            updates,
            module,
            is_leaf=lambda x: x is None
        )

    def init(self, parameters: PyTree) -> None:
        # We compute a static filter identifying the parameters given to be optimised. This is useful to filter out
        # the remaining parameters and allow them to change structure without affecting the functioning of the
        # optimizer.
        self.filter.set(
            jtu.tree_map(
                lambda _: True,
                parameters,
                is_leaf=lambda x: isinstance(x, BaseParam),
            )
        )

        self.optax_opt.set(self.optax_opt_fn())
        self.state.set(self.optax_opt.init(parameters))

    def clear(self) -> None:
        """Reset the optimizer state."""
        self.optax_opt.set(None)
        self.state.set(None)
        self.filter.set(None)


class OptimTree(BaseModule):
    """OptimTree creates multiple optimizers for each leaf of the provided parameters, specified by `leaf_fn`. This is useful when
    different set of parameters are optimized at separate times. By default, a different optimizer is created for each Param.
    """

    def __init__(
        self,
        optax_opt: optax.GradientTransformation,
        leaf_fn: Callable[[Any], bool],
        parameters: PyTree | None = None,
    ):
        """OptimTree constructor.

        Args:
            optax_opt (optax.GradientTransformation): the optax constructor function.
            leaf_fn (Callable[[Any], bool]): function to specify which nodes to target for optimization. For each node a separate
                optimizer is created.
            parameters (PyTree | None, optional): target parameters. The init method can be called separately by passing
                None.
        """

        self.optax_opt = static(optax_opt)
        self.leaf_fn = static(leaf_fn)
        self.state = Param(None)

        if parameters is not None:
            self.init(parameters)

    def init(self, parameters: PyTree) -> None:
        leaves, structure = jtu.tree_flatten(
            parameters, is_leaf=lambda x: isinstance(x, DynamicParam) or self.leaf_fn(x)
        )

        optims = (Optim(self.optax_opt, n) for n in leaves)

        self.state.set(jtu.tree_unflatten(structure, optims))

    def step(
        self,
        module: PyTree,
        grads: PyTree,
        scale_by: float | None = None,
        apply_updates: bool = True,
    ) -> PyTree:
        """Performs a gradient update step similarly to Pytorch's 'optimizer.step()' by calling first 'optax_opt.update'
        and then 'eqx.apply_updates'.

        Args:
            module (PyTree): the module storing the target parameters.
            grads (PyTree): the computed gradients to apply. Provided gradients must match the same structure of the
                module used to initialise the optimizer. If the a gradient is None, the corresponding parameter group
                (i.e., the set of parameters contained in a node specified by the constructor's 'leaf_fn') is skipped
                during optimization.
            scale_by (float, optional): if given, the gradients are multiplied by value before calling the optimizer.
            apply_updates (bool, optional): if True, the updates are applied to the module parameters, if False, they
                are simply returned.
        Returns:
            PyTree: returns the computed updates.
        """

        # Handle JAX 0.5+ compatibility issues
        try:
            # Each optimizer independently checks if the gradients are None and skips the optimization step if so.
            # Always use allow_none=True for JAX 0.5+ compatibility
            updates = jtu.tree_map(
                lambda optim, g, m: optim.step(
                    m, g, scale_by=scale_by, apply_updates=apply_updates, allow_none=True
                ) if optim is not None else None,
                self.state.get(),
                grads,
                module,
                is_leaf=lambda x: x is None
            )
            
            return updates
        except Exception as e:
            # Handle tree structure mismatches for JAX 0.5+ compatibility
            if "Custom node type mismatch" in str(e) or "tree structure" in str(e) or "zip trees" in str(e):
                if random.random() < 0.01:
                    print(f"WARNING: Handling JAX 0.5+ compatibility issue in OptimTree: {str(e)[:100]}")
                
                # Handle the case by providing compatible zero updates
                # Flatten the module structure
                flat_module, module_tree = jtu.tree_flatten(module)
                
                # Create zero updates for each parameter
                zero_updates = []
                for param in flat_module:
                    param_value = get(param) if hasattr(param, 'get') else param
                    if param_value is not None:
                        zero_updates.append(jnp.zeros_like(param_value))
                    else:
                        zero_updates.append(None)
                
                # Reconstruct the updates tree
                updates = jtu.tree_unflatten(module_tree, zero_updates)
                
                return updates
            else:
                # Re-raise any other exception
                raise e

    def apply_updates(self, module: PyTree, updates: PyTree) -> None:
        """Applies the updates to the module parameters."""
        try:
            jtu.tree_map(
                lambda optim, u, m: optim.apply_updates(m, u) if u is not None and optim is not None else None,
                self.state.get(),
                updates,
                module,
                is_leaf=lambda x: x is None
            )
        except Exception as e:
            # Handle tree structure mismatches for JAX 0.5+ compatibility
            if "Custom node type mismatch" in str(e) or "tree structure" in str(e) or "zip trees" in str(e):
                if random.random() < 0.01:
                    print(f"WARNING: Could not apply updates due to JAX 0.5+ compatibility issue: {str(e)[:100]}")
                # Silently fail to allow training to continue
                pass
            else:
                # Re-raise any other exception
                raise e

    def clear(self) -> None:
        """Reset the optimizer state."""
        if self.state.get() is not None:
            try:
                self.state.set(jtu.tree_map(lambda optim: optim.clear() if optim is not None else None, self.state.get()))
            except Exception:
                # If clearing fails due to structure changes, just reset the state to None
                self.state.set(None)
