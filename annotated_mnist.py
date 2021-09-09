# %% [markdown]
# # Annotated MNIST
#
# This tutorial demonstrates how to construct a simple convolutional neural
# network (CNN) using the [Flax](https://flax.readthedocs.io) Linen API and train
# the network for image classification on the MNIST dataset.
#
# Note: This notebook is based on Flax's official
# [MNIST Example](https://github.com/google/flax/tree/main/examples/mnist).
# If you see any changes between the two feel free to create a
# [pull request](https://github.com/google/flax/compare)
# to synchronize this Colab with the actual example.
# %% [markdown]
# ## 1. Imports
#
# Import JAX, [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html),
# Flax, ordinary NumPy, and HuggingFace Datasets. Flax can use any data-loading
# pipeline and this example was modified to use the HuggingFace datasets library

# %%
import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import datasets                        # Huggingface datasets for MNIST

datasets.logging.set_verbosity_error()

# %% [markdown]
# ## 2. Define network
#
# Create a convolutional neural network with the Linen API by subclassing
# [`Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#core-module-abstraction).
# Because the architecture in this example is relatively simple—you're just
# stacking layers—you can define the inlined submodules directly within the
# `__call__` method and wrap it with the
# [`@compact`](https://flax.readthedocs.io/en/latest/flax.linen.html#compact-methods)
# decorator.

# %%
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


# %% [markdown]
# ## 3. Define loss
#
# Define a cross-entropy loss function using just
# [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html)
# that takes the model's logits and label vectors and returns a scalar loss. The
# labels can be one-hot encoded with
# [`jax.nn.one_hot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.one_hot.html),
# as demonstrated below.
#
# Note that for demonstration purposes, we return `nn.log_softmax()` from
# the model and then simply multiply these (normalized) logits with the labels. In our
# `examples/mnist` folder we actually return non-normalized logits and then use
# `optax.softmax_cross_entropy()` to compute the loss, which has the same result.

# %%
def cross_entropy_loss(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


# %% [markdown]
# ## 4. Metric computation
#
# For loss and accuracy metrics, create a separate function:

# %%
def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


# %% [markdown]
# ## 5. Loading data
#
# Define a function that loads and prepares the MNIST dataset and converts the
# samples to floating-point numbers.

# %%
def get_datasets():
    """Load MNIST train and test datasets into memory."""
    mnist = datasets.load_dataset("mnist")
    mnist.set_format("numpy")
    train_ds = {
        "image": jnp.float32(mnist["train"]["image"][..., None]) / 255.0,
        "label": mnist["train"]["label"],
    }
    test_ds = {
        "image": jnp.float32(mnist["test"]["image"][..., None]) / 255.0,
        "label": mnist["test"]["label"],
    }
    return train_ds, test_ds


# %% [markdown]
# ## 6. Create train state
#
# A common pattern in Flax is to create a single dataclass that represents the
# entire training state, including step number, parameters, and optimizer state.
#
# Also adding optimizer & model to this state has the advantage that we only need
# to pass around a single argument to functions like `train_step()` (see below).
#
# Because this is such a common pattern, Flax provides the class
# [flax.training.train_state.TrainState](https://flax.readthedocs.io/en/latest/flax.training.html#train-state)
# that serves most basic usecases. Usually one would subclass it to add more data
# to be tracked, but in this example we can use it without any modifications.

# %%
def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


# %% [markdown]
# ## 7. Training step
#
# A function that:
#
# - Evaluates the neural network given the parameters and a batch of input images
#   with the
#   [`Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply)
#   method.
# - Computes the `cross_entropy_loss` loss function.
# - Evaluates the loss function and its gradient using
#   [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad).
# - Applies a
#   [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions)
#   of gradients to the optimizer to update the model's parameters.
# - Computes the metrics using `compute_metrics` (defined earlier).
#
# Use JAX's [`@jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit)
# decorator to trace the entire `train_step` function and just-in-time compile
# it with [XLA](https://www.tensorflow.org/xla) into fused device operations
# that run faster and more efficiently on hardware accelerators.

# %%
@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = CNN().apply({"params": params}, batch["image"])
        loss = cross_entropy_loss(logits=logits, labels=batch["label"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch["label"])
    return state, metrics


# %% [markdown]
# ## 8. Evaluation step
#
# Create a function that evaluates your model on the test set with
# [`Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply)

# %%
@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({"params": params}, batch["image"])
    return compute_metrics(logits=logits, labels=batch["label"])


# %% [markdown]
# ## 9. Train function
#
# Define a training function that:
#
# - Shuffles the training data before each epoch using
#   [`jax.random.permutation`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.permutation.html)
#   that takes a PRNGKey as a parameter (check the
#   [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG)).
# - Runs an optimization step for each batch.
# - Retrieves the training metrics from the device with `jax.device_get` and
#   computes their mean across each batch in an epoch.
# - Returns the optimizer with updated parameters and the training loss and
#   accuracy metrics.

# %%
def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print(
        "train epoch: %d, loss: %.4f, accuracy: %.2f"
        % (epoch, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"] * 100)
    )

    return state


# %% [markdown]
# ## 10. Eval function
#
# Create a model evaluation function that:
#
# - Retrieves the evaluation metrics from the device with `jax.device_get`.
# - Copies the metrics
#   [data stored](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables)
#   in a JAX
#   [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions).

# %%
def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary["loss"], summary["accuracy"]


# %% [markdown]
# ## 11. Download data

# %%
train_ds, test_ds = get_datasets()

# %% [markdown]
# ## 12. Seed randomness
#
# - Get one
#   [PRNGKey](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax.random.PRNGKey)
#   and
#   [split](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split)
#   it to get a second key that you'll use for parameter initialization. (Learn
#   more about
#   [PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables)
#   and
#   [JAX PRNG design](https://github.com/google/jax/blob/main/design_notes/prng.md).)

# %%
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# %% [markdown]
# ## 13. Initialize train state
#
# Remember that function initializes both the model parameters and the optimizer
# and puts both into the training state dataclass that is returned.

# %%
learning_rate = 0.1
momentum = 0.9


# %%
state = create_train_state(init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

# %% [markdown]
# ## 14. Train and evaluate
#
# Once the training and testing is done after 10 epochs, the output should show that your model was able to achieve approximately 99% accuracy.

# %%
num_epochs = 10
batch_size = 32


# %%
for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
    # Evaluate on the test set after each training epoch
    test_loss, test_accuracy = eval_model(state.params, test_ds)
    print(
        " test epoch: %d, loss: %.2f, accuracy: %.2f"
        % (epoch, test_loss, test_accuracy * 100)
    )

# %% [markdown]
# Congrats! You made it to the end of the annotated MNIST example. You can revisit
# the same example, but structured differently as a couple of Python modules, test
# modules, config files, another Colab, and documentation in Flax's Git repo:
#
# https://github.com/google/flax/tree/main/examples/mnist
#
