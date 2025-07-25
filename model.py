from typing import List
import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp

class WidthModel(eqx.Module):

    q_layers_conv: List
    amplitude_conv: List
    dense: List

    def __init__(self, key=None):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        
        self.q_layers_conv = [
            # Give (61, 20)
            nn.Conv2d(1, 1, kernel_size=(3, 7), padding="SAME", padding_mode="ZEROS", key=k1),
            nn.Conv2d(1, 1, kernel_size=(3, 3), padding="SAME", padding_mode="ZEROS", key=k2),
        ]
        self.amplitude_conv = [
            nn.Conv1d(1, 1, kernel_size=20, padding="SAME", key=k3)
        ]
        self.dense = [
            nn.Linear((61 * 20) + (20), 40, key=k4),
            nn.Linear(40, 1, key=k5),
        ]

        self.dense = [
            nn.Linear(40, 1, key=k1)
        ]

    def __call__(self, q_layers: jax.Array, amplitude: jax.Array) -> jax.Array:
        # q_layers = jnp.expand_dims(q_layers, axis=0)  # Add batch dimension
        # for layer in self.q_layers_conv:
        #     q_layers = layer(q_layers)
        #     q_layers = jax.nn.sigmoid(q_layers)

        # x = jnp.concatenate([q_layers.flatten(), amplitude.flatten()], axis=-1)
        # for layer in self.dense[:-1]:
        #     x = layer(x)
        #     x = jax.nn.sigmoid(x)
        # x = self.dense[-1](x)

        x = self.dense[-1](amplitude)


        return x