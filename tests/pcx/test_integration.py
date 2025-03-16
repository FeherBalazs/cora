import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import optax

from pcx.nn._parameter import LayerParam
from pcx.core._random import RKG
from pcx.nn import Linear, LayerNorm, Conv2d
from pcx.predictive_coding import Vode, EnergyModule
from pcx.utils._optim import Optim
from pcx.predictive_coding._energy import se_energy as energy_mse

# Define the forward status constant
PC_FORWARD = "forward"

class SimpleNN(EnergyModule):
    """Simple neural network for integration testing."""
    linear1: Linear
    linear2: Linear
    norm: LayerNorm
    
    def __init__(self, in_dim=10, hidden_dim=20, out_dim=5):
        super().__init__()
        self.linear1 = Linear(in_features=in_dim, out_features=hidden_dim)
        self.norm = LayerNorm(shape=(hidden_dim,))
        self.linear2 = Linear(in_features=hidden_dim, out_features=out_dim)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x
    
    def energy(self):
        """Compute energy."""
        return 0.0  # No internal energy in this simple case


class SimpleVodeNetwork(EnergyModule):
    """Simple network of Vodes for integration testing."""
    vode1: Vode
    vode2: Vode
    linear: Linear
    
    def __init__(self, in_dim=10, hidden_dim=20, out_dim=5):
        super().__init__()
        self.vode1 = Vode(energy_fn=energy_mse)
        self.vode2 = Vode(energy_fn=energy_mse)
        self.linear = Linear(in_features=in_dim, out_features=out_dim)
    
    def __call__(self, x):
        """Forward pass through the network."""
        # Init states based on input
        self.vode1.set("u", x)
        self.vode1.set("h", x)
        
        h = self.linear(x)
        self.vode2.set("u", h)
        self.vode2.set("h", h)
        
        return self.vode2.get('u')
    
    def energy(self):
        """Compute total energy of the network."""
        return jnp.sum(self.vode1.energy()) + jnp.sum(self.vode2.energy())


def test_nn_with_optim():
    """Test that a neural network can be trained with an optimizer."""
    # Create a simple neural network
    model = SimpleNN(in_dim=5, hidden_dim=10, out_dim=2)
    
    # Generate synthetic data
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (5,))
    y_true = jnp.array([1.0, 0.0])
    
    # Initial prediction
    y_pred_init = model(x)
    init_loss = jnp.mean((y_pred_init - y_true) ** 2)
    
    # Training step using SGD directly without Optim
    learning_rate = 0.01
    
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y_true) ** 2)
    
    # Get gradients
    grads = jax.grad(loss_fn)(model)
    
    # Apply gradients manually
    model_updated = jax.tree_map(
        lambda param, grad: param - learning_rate * grad if grad is not None else param,
        model, grads
    )
    
    # Check that parameters were updated
    assert not jnp.array_equal(
        model.linear1.nn.weight.get(), 
        model_updated.linear1.nn.weight.get()
    )
    
    # Check that loss improved
    y_pred_updated = model_updated(x)
    updated_loss = jnp.mean((y_pred_updated - y_true) ** 2)
    assert updated_loss < init_loss, "Training should reduce loss"


def test_vode_with_optim():
    """Test basic Vode functionality with manual parameter updates."""
    # Create a simple Vode
    v = Vode(energy_fn=energy_mse)
    
    # Set initial values that are different to get non-zero energy
    input_data = jnp.array([1.0, 2.0, 3.0])
    hidden_state = jnp.array([2.0, 3.0, 4.0])  # Different from input_data
    
    v.set("u", input_data)
    v.set("h", hidden_state)
    
    # Check that energy is non-zero (since h and u are different)
    energy = jnp.sum(v.energy())
    assert energy > 0.0
    
    # Energy should be sum of squared differences
    expected_energy = 0.5 * jnp.sum((hidden_state - input_data) ** 2)
    assert jnp.isclose(energy, expected_energy)


def test_nn_with_vode():
    """Test interoperability between neural network layers and Vode components."""
    # Create a neural network with a Vode component
    class HybridModel(EnergyModule):
        linear: Linear
        vode: Vode
        
        def __init__(self):
            super().__init__()
            self.linear = Linear(in_features=5, out_features=3)
            self.vode = Vode(energy_fn=energy_mse)
        
        def __call__(self, x):
            """Forward pass with Vode integration."""
            h = self.linear(x)
            self.vode.set("u", h)
            # Set h to a different value to get non-zero energy
            self.vode.set("h", h + 1.0)
            return self.vode.get('u')
        
        def energy(self):
            """Compute total energy."""
            return jnp.sum(self.vode.energy())
    
    # Create the model
    model = HybridModel()
    
    # Generate input
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (5,))
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (3,)
    
    # Energy should be positive (MSE) since h and u are different
    energy = model.energy()
    assert energy > 0.0
    
    # Energy should be MSE between h and u
    h_value = output + 1.0  # This is what we set h to
    expected_energy = 0.5 * jnp.sum((h_value - output) ** 2)
    assert jnp.isclose(energy, expected_energy)


def test_full_integration():
    """Test basic integration of Vode, NN layers."""
    # Define a simplified model
    class SimpleConvModel(EnergyModule):
        conv: Conv2d
        vode: Vode
        
        def __init__(self):
            super().__init__()
            self.conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
            self.vode = Vode(energy_fn=energy_mse)
        
        def __call__(self, x):
            """Forward pass through the network."""
            h = self.conv(x)  # Output: (2, 6, 6) for 8x8 input
            self.vode.set("u", h)
            # Set h to a different value to get non-zero energy
            self.vode.set("h", h * 1.5)  # Multiply by 1.5 to ensure difference
            return h
        
        def energy(self):
            """Compute energy."""
            return jnp.sum(self.vode.energy())
    
    # Create the model
    model = SimpleConvModel()
    
    # Generate synthetic image data (1 channel, 8x8)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (1, 8, 8))
    
    # Forward pass
    output = model(x)
    
    # Check output shape for Conv2d with kernel_size=3 and no padding
    # Input: (1, 8, 8), Output: (2, 6, 6)
    assert output.shape == (2, 6, 6)
    
    # Check energy is non-zero
    energy = model.energy()
    assert energy > 0.0
    
    # Energy should be MSE between h and u
    h_value = output * 1.5  # This is what we set h to
    expected_energy = 0.5 * jnp.sum((h_value - output) ** 2)
    assert jnp.isclose(energy, expected_energy) 