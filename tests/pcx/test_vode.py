import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pcx
from pcx.predictive_coding import Vode, STATUS
from pcx.core._random import RKG

def test_vode_init():
    """Test that Vode initializes correctly with default parameters."""
    v = Vode()
    assert v is not None
    assert hasattr(v, 'h')
    assert hasattr(v, 'u')
    
def test_vode_call_without_input():
    """Test that Vode.__call__ returns None when called without input."""
    v = Vode()
    result = v(None)
    assert result is None
    
def test_vode_call_with_input():
    """Test that Vode.__call__ processes input correctly."""
    v = Vode()
    input_data = jnp.array([1.0, 2.0, 3.0])
    result = v(input_data)
    assert result is not None
    assert jnp.array_equal(v.get("h"), input_data)
    
def test_vode_set_get():
    """Test that Vode.set and Vode.get work correctly."""
    v = Vode()
    test_array = jnp.array([4.0, 5.0, 6.0])
    v.set("h", test_array)
    result = v.get("h")
    assert jnp.array_equal(result, test_array)
    
def test_vode_energy():
    """Test that Vode.energy computes energy correctly."""
    v = Vode()
    h_value = jnp.array([1.0, 2.0, 3.0])
    u_value = jnp.array([1.5, 2.5, 3.5])
    v.set("h", h_value)
    v.set("u", u_value)
    # Default energy function is se_energy (squared error)
    energy = v.energy()
    expected_energy = jnp.sum((h_value - u_value) ** 2)
    assert jnp.allclose(energy, expected_energy)
    
def test_vode_custom_energy_fn():
    """Test that Vode works with a custom energy function."""
    def custom_energy(vode, rkg):
        h = vode.get("h")
        u = vode.get("u")
        return jnp.sum(jnp.abs(h - u))  # L1 norm instead of L2
        
    v = Vode(energy_fn=custom_energy)
    h_value = jnp.array([1.0, 2.0, 3.0])
    u_value = jnp.array([1.5, 2.5, 3.5])
    v.set("h", h_value)
    v.set("u", u_value)
    energy = v.energy()
    expected_energy = jnp.sum(jnp.abs(h_value - u_value))
    assert jnp.allclose(energy, expected_energy)
    
def test_vode_ruleset():
    """Test that Vode rulesets work correctly."""
    # Create a ruleset that initializes h to zeros
    def init_zeros(node, key, value, rkg):
        return jnp.zeros_like(value)
    
    ruleset = {STATUS.INIT: ["h <- u:init_zeros"]}
    tforms = {"init_zeros": init_zeros}
    
    v = Vode(ruleset=ruleset, tforms=tforms)
    
    # Test INIT status
    input_data = jnp.array([1.0, 2.0, 3.0])
    with pcx.utils.step(v, STATUS.INIT):
        result = v(input_data)
    
    # h should be zeros, not input_data
    assert jnp.array_equal(v.get("h"), jnp.zeros_like(input_data))
    
def test_vode_forward_and_backward():
    """Test a simple forward and backward pass with Vode."""
    v = Vode()
    
    # Forward pass
    input_data = jnp.array([1.0, 2.0, 3.0])
    result = v(input_data)
    assert jnp.array_equal(result, input_data)
    
    # Simulate a backward pass by modifying u
    target = jnp.array([1.5, 2.5, 3.5])
    v.set("u", target)
    
    # Energy should reflect the difference
    energy = v.energy()
    expected_energy = jnp.sum((input_data - target) ** 2)
    assert jnp.allclose(energy, expected_energy) 