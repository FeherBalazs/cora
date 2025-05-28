__all__ = ["zero_energy", "se_energy", "ce_energy", "regularized_plus_se_energy"]


import jax
import jax.numpy as jnp

from ..core._random import RKG, RandomKeyGenerator


########################################################################################################################
#
# Energy
#
# Collection of the most common energy functions used in predictive coding.
#
########################################################################################################################


# Core #################################################################################################################


def zero_energy(vode, rkg: RandomKeyGenerator = RKG):
    """used to unconstrain the value of a vode from its prior distribution (i.e., input)."""
    return jax.numpy.zeros((1,))


def se_energy(vode, rkg: RandomKeyGenerator = RKG):
    """Squared error energy function derived from a Gaussian distribution."""
    e = vode.get("h") - vode.get("u")
    return 0.5 * (e * e)


def ce_energy(vode, rkg: RandomKeyGenerator = RKG):
    """Cross entropy energy function derived from a categorical distribution."""
    return -(vode.get("h") * jax.nn.log_softmax(vode.get("u")))


def regularized_plus_se_energy(vode, rkg: RandomKeyGenerator, l1_coeff: float = 0.001, l2_coeff: float = 0.001):
    """
    Combines the Vode's inherent Squared Error energy (between its h and u)
    with L1 and L2 regularization penalties for its hidden state 'h'.
    """
    h_val = vode.get("h")
    u_val = vode.get("u") # u is the prediction/target for h from the layer below

    # Calculate original SE energy contribution
    # This assumes 'u' is a meaningful target for 'h' in intermediate vodes during optimization
    if h_val is None:
        raise ValueError("Vode 'h' is None during energy calculation.")
    
    se_term_elementwise = jnp.zeros_like(h_val) # Default to zero if u is not set
    if u_val is not None:
        se_error = h_val - u_val
        se_term_elementwise = 0.5 * jnp.square(se_error)
    # else:
        # If u_val is None for an intermediate Vode, it means it has no "target" for its h from below,
        # which would be unusual if it's meant to have a local SE term.
        # Or, pcx.predictive_coding.se_energy might handle this by erroring or returning 0.
        # For safety, if u is not there, we assume its SE contribution is 0 for this Vode.

    # Calculate regularization penalties for 'h'
    _l1_coeff = l1_coeff if l1_coeff is not None else 0.0
    _l2_coeff = l2_coeff if l2_coeff is not None else 0.0

    l1_penalty_elementwise = _l1_coeff * jnp.abs(h_val)
    l2_penalty_elementwise = _l2_coeff * jnp.square(h_val) # l2_coeff * h^2

    return se_term_elementwise + l1_penalty_elementwise + l2_penalty_elementwise
