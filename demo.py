# %% Imports

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# %% JAX is like numpy

jnp.ones((4,4))

# %%

a = jnp.asarray(np.random.normal(size=(2, 4)))

# %% gelu

def gelu(x):
    cdf = 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * (x ** 3))))
    return x * cdf

# %%
# %%
# %% 

x = jnp.linspace(-7, 7, 200)
plt.plot(x, gelu(x))

# %% Taking gradients
# %%
# %%
# %% Vectorization

def printing_gelu(x):
    print('shape of x is', x.shape)
    return gelu(x)

# %%
# %%
# %% Linear Regression

xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * 2 - 1 + noise
plt.scatter(xs, ys)

# %%

initial_weight = jnp.asarray(np.random.normal())
initial_bias = jnp.asarray(np.random.normal())

plt.scatter(xs, ys)
plt.plot(xs, initial_weight * xs + initial_bias, color='red')

# %%
# %%
# %%

def loss(weight, bias, x, y):
    ...

# %%
# %%
# %% Structured objects

from flax.struct import dataclass

@dataclass
class WeightBiasPair:
    weight: jnp.ndarray
    bias: jnp.ndarray

def loss(params, x, y):
    ...

...
plt.scatter(xs, ys)
plt.plot(xs, initial_params.weight * xs + initial_params.bias, color='red')
loss(initial_params, xs, ys)

# %%
# %%
# %% Gradients and parameter updates
# %%
# %%
# %% Training loop and JIT
# %%
# %%
# %% Learning rate adjustment

def train(params, x, y, lr):
    for _ in range(100):
        ...
    final_loss = loss(params, x, y)
    return final_loss, params


final_loss, params = train(initial_params, xs, ys, 0.005)

plt.scatter(xs, ys)
plt.plot(xs, params.weight * xs + params.bias, color='red')

# %%
# %%
# %% JVP
# %%
# %%
# %% Optax

import optax
tx = optax.scale(-0.005)
state = tx.init(initial_params)

params = initial_params
...

# %%
# %%
