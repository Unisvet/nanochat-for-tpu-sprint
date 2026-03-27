import jax
import jax.numpy as jnp

y = jnp.array([1, 2, 0])
vocab_size = 5
one_hot = jax.nn.one_hot(y, vocab_size)
print(one_hot)
print(one_hot.shape)
