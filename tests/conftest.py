"""
Common fixtures for PCX tests.
"""
import os
import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pcx.predictive_coding import Vode, EnergyModule
from pcx.predictive_coding._energy import se_energy as energy_mse
from pcx.nn import Linear
from pcx.core._random import RKG

# Define the forward status constant
PC_FORWARD = "forward"

# Configure pytest
def pytest_configure(config):
    # Register custom marks
    config.addinivalue_line("markers", "slow: mark test as slow (skipped by default)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    
    # Configure other pytest settings if needed
    pass

# Fixtures available to all tests
@pytest.fixture
def small_test_array():
    """Return a small test array for simple tests."""
    import jax.numpy as jnp
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def test_matrix():
    """Return a test matrix for testing matrix operations."""
    import jax.numpy as jnp
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

@pytest.fixture
def batch_input():
    """Return a batch of inputs for testing batch processing."""
    import jax.numpy as jnp
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

@pytest.fixture
def batch_target():
    """Return a batch of targets for testing supervised learning."""
    import jax.numpy as jnp
    return jnp.array([[0.0, 1.0], [1.0, 0.0]])

@pytest.fixture(scope="session")
def random_seed():
    """Return a random seed for deterministic tests."""
    return 42

@pytest.fixture(scope="function")
def rng_key():
    """Return a fixed RNG key for deterministic tests."""
    return jax.random.PRNGKey(42)

@pytest.fixture(scope="function")
def random_key_generator():
    """Return a random key generator initialized with a fixed seed."""
    return RKG(42)

@pytest.fixture(scope="function")
def simple_vode():
    """Return a simple Vode for testing."""
    vode = Vode(
        energy_fn=energy_mse,
        ruleset={"FORWARD": [{"status": PC_FORWARD, "u": "u"}]}
    )
    return vode

@pytest.fixture(scope="function")
def simple_energy_module():
    """Return a simple EnergyModule for testing."""
    class SimpleEnergyModule(EnergyModule):
        param1: jnp.ndarray
        param2: jnp.ndarray
        
        def __init__(self):
            super().__init__()
            self.param1 = jnp.array([1.0, 2.0, 3.0])
            self.param2 = jnp.array([4.0, 5.0])
        
        def energy(self, status=PC_FORWARD):
            """Compute a simple energy value."""
            return jnp.sum(self.param1) + jnp.sum(self.param2)
    
    return SimpleEnergyModule()

@pytest.fixture(scope="function")
def linear_layer():
    """Return a linear layer with deterministic weights and biases."""
    linear = Linear(in_features=3, out_features=2)
    # Set weights and biases to specific values for deterministic testing
    linear.nn.weight.set(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    linear.nn.bias.set(jnp.array([0.1, 0.2]))
    return linear

@pytest.fixture(scope="function")
def sample_batch():
    """Return a sample batch of data for testing."""
    x = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    y = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ])
    return x, y 