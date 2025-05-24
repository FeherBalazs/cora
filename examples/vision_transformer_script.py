import functools

import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax  # https://github.com/deepmind/optax

# We'll use PyTorch to load the dataset.
import torch
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Array, Float, PRNGKeyArray

# Hyperparameters
lr = 0.001
dropout_rate = 0.0
beta1 = 0.9
beta2 = 0.999
batch_size = 64
patch_size = 4
num_patches = 64
num_steps = 50000  # Reduced for faster execution in script form, adjust as needed
image_size = (32, 32, 3)
embedding_dim = 64
hidden_dim = 256
num_heads = 1
num_layers = 6
height, width, channels = image_size
num_classes = 10

# Load CIFAR10 dataset
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    "CIFAR",
    train=True,
    download=True,
    transform=transform_train,
)

test_dataset = torchvision.datasets.CIFAR10(
    "CIFAR",
    train=False,
    download=True,
    transform=transform_test,
)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# Patch embeddings layer
class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear  # Corrected type annotation
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)

        return x

# Attention block
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "num_patches embedding_dim"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_patches embedding_dim"]:
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x, key=jr.fold_in(key, 0)) # Added key for attention

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, inference=not enable_dropout, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout, key=key2)

        x = x + input_x

        return x

# Vision Transformer model
class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.Sequential
    num_layers: int

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
    ):
        key1, key2, key3, key_att_blocks, key_mlp = jr.split(key, 5) # Renamed key4 to key_att_blocks, key5 to key_mlp

        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)

        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))

        self.cls_token = jr.normal(key3, (1, embedding_dim))

        self.num_layers = num_layers
        
        # Create keys for each attention block
        attention_block_keys = jr.split(key_att_blocks, self.num_layers)
        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, attention_key) # Corrected typo
            for attention_key in attention_block_keys # Corrected variable name and iteration
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=key_mlp), # Use key_mlp
            ]
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_classes"]:
        x = self.patch_embedding(x)  # Output shape for single image: (num_patches, embedding_dim)

        # Prepend the single CLS token (self.cls_token shape is (1, embedding_dim))
        x = jnp.concatenate((self.cls_token, x), axis=0) # Shape: (1 + num_patches, embedding_dim)

        x += self.positional_embedding[
            : x.shape[0] # Slice positional embedding to match sequence length (1 + num_patches)
        ]

        dropout_key, *attention_keys = jr.split(key, num=self.num_layers + 1)

        x = self.dropout(x, inference=not enable_dropout, key=dropout_key)

        for block, attention_key_loop in zip(self.attention_blocks, attention_keys): # Renamed attention_key to attention_key_loop
            x = block(x, enable_dropout, key=attention_key_loop)

        # When __call__ processes a single instance (due to outer vmap):
        # x shape is (1 + num_patches, embedding_dim)
        cls_output = x[0]  # Select the CLS token. Shape: (embedding_dim,)
        output = self.mlp(cls_output) # Apply MLP. Shape: (num_classes,)

        return output

# Gradient computation and training step
@eqx.filter_value_and_grad
def compute_grads(
    model: VisionTransformer, images: jnp.ndarray, labels: jnp.ndarray, key: PRNGKeyArray
):
    # Keys for vmap need to be handled correctly.
    # Assuming one key per batch item for the model call.
    batch_keys = jr.split(key, images.shape[0])
    logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, batch_keys)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


@eqx.filter_jit
def step_model(
    model: VisionTransformer,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: PRNGKeyArray, # Single key for the step
):
    loss, grads = compute_grads(model, images, labels, key) # Pass the single key
    updates, new_state = optimizer.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


def train(
    model: VisionTransformer,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    data_loader: torch.utils.data.DataLoader,
    num_steps: int,
    print_every: int = 1000,
    key: PRNGKeyArray = None, # Default to None
):
    if key is None:
        key = jr.PRNGKey(0) # Default key if not provided

    losses = []
    data_iter = iter(data_loader) # Use a direct iterator

    for step in range(num_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration: # Reset iterator if dataset is exhausted
            data_iter = iter(data_loader)
            images, labels = next(data_iter)


        images = images.numpy()
        labels = labels.numpy()

        step_key, key = jr.split(key) # Split key for the current step

        model, state, loss = step_model(
            model, optimizer, state, images, labels, step_key # Use step_key
        )

        losses.append(loss.item()) # .item() to get scalar

        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step: {step}/{num_steps}, Loss: {loss.item()}.")

    return model, state, losses

# Main execution block
if __name__ == "__main__":
    key = jr.PRNGKey(2003)
    model_key, train_key, eval_key = jr.split(key, 3)


    model = VisionTransformer(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        key=model_key, # Use model_key
    )

    optimizer = optax.adamw(
        learning_rate=lr,
        b1=beta1,
        b2=beta2,
    )

    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    print("Starting training...")
    model, state, losses = train(model, optimizer, state, trainloader, num_steps, key=train_key) # Use train_key
    print("Training finished.")

    # Evaluation
    print("Starting evaluation...")
    accuracies = []
    
    eval_data_iter = iter(testloader)
    num_eval_batches = len(test_dataset) // batch_size

    for _ in range(num_eval_batches): # Iterate a fixed number of batches
        try:
            images, labels = next(eval_data_iter)
        except StopIteration:
            # This shouldn't happen if num_eval_batches is correct
            print("Warning: Test loader exhausted prematurely.")
            break 

        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Create keys for each image in the batch for the model call during evaluation
        batch_eval_keys = jr.split(eval_key, images_np.shape[0])
        eval_key, _ = jr.split(eval_key) # Consume the key by splitting

        logits = jax.vmap(model, in_axes=(0, None, 0))(
            images_np, False, batch_eval_keys
        )

        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels_np)
        accuracies.append(accuracy.item()) # .item() to get scalar

    if accuracies:
        avg_accuracy = np.mean(accuracies) * 100
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
    else:
        print("No accuracies calculated.")
    print("Evaluation finished.") 