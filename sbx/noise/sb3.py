"""Colored noise implementations for Stable Baselines3"""

import numpy as np
import torch as th
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from stable_baselines3.common.noise import ActionNoise

from .cnrl import ColoredNoiseProcess


class ColoredActionNoise(ActionNoise):
    def __init__(self, beta, sigma, seq_len, action_dim=None, rng=None):
        """Action noise from a colored noise process.

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__()
        assert (action_dim is not None) == np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        self.sigma = np.full(action_dim or len(beta), sigma) if np.isscalar(sigma) else np.asarray(sigma)

        if np.isscalar(beta):
            self.beta = beta
            self.gen = ColoredNoiseProcess(beta=self.beta, scale=self.sigma, size=(action_dim, seq_len), rng=rng)
        else:
            self.beta = np.asarray(beta)
            self.gen = [ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                        for b, s in zip(self.beta, self.sigma)]

    def __call__(self) -> np.ndarray:
        return self.gen.sample() if np.isscalar(self.beta) else np.asarray([g.sample() for g in self.gen])

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self.beta}, sigma={self.sigma})"


class PinkActionNoise(ColoredActionNoise):
    def __init__(self, sigma, seq_len, action_dim, rng=None):
        """Action noise from a pink noise process.

        Parameters
        ----------
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled pink noise signals. If sampled for longer than `seq_len` steps, a new
            pink noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, sigma, seq_len, action_dim, rng)


import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

from jax import numpy as jnp

        
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.math

class MyMultivariateNormalDiag(tfd.Distribution):
    def __init__(self, loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, name="MultivariateNormalDiag"):
        parameters = dict(locals())
        #with tfp.util.deferred_dependencies.defer_dependencies():
        self._loc = jnp.zeros_like(scale_diag) if loc is None else jnp.asarray(loc)
        self._scale_diag = jnp.asarray(scale_diag)
        
        if self._loc.shape != self._scale_diag.shape:
            raise ValueError(f"Shape mismatch: loc {self._loc.shape} vs scale_diag {self._scale_diag.shape}")
        
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])
        self._event_shape_ = self._loc.shape[-1:]
        
        super().__init__(
            dtype=self._loc.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @property
    def loc(self):
        return self._loc

    @property
    def scale_diag(self):
        return self._scale_diag

    def _batch_shape(self):
        return self._batch_shape_

    def _event_shape(self):
        return self._event_shape_

    def _sample_n(self, sample_shape, seed):
        key = jax.random.split(seed)[0]
        if type(sample_shape) is not tuple:
            sample_shape = (sample_shape,)
        eps = jax.random.normal(key, shape=sample_shape + self._batch_shape_ + self._event_shape_)
        return self._loc + eps * self._scale_diag

    def _log_prob(self, value):
        var = jnp.square(self._scale_diag)
        log_scale = jnp.log(self._scale_diag)
        return -0.5 * (jnp.square(value - self._loc) / var + 2. * log_scale + jnp.log(2. * jnp.pi)).sum(axis=-1)

    def _mean(self):
        return self._loc

    def _stddev(self):
        return self._scale_diag

    def set_mean_and_scale_diag(self, mean, scale_diag):
        self._loc = mean
        self._scale_diag = scale_diag
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])

    def _entropy(self):
        return jnp.sum(
            jnp.log(self._scale_diag * jnp.sqrt(2. * jnp.pi * jnp.e)),
            axis=-1
        )

    def mode(self):
        return self._loc


class ColoredNoiseDist(tfd.Distribution):
    def __init__(self, beta=1.0, seq_len=100., key=None, loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, name="MultivariateNormalDiag"):
        parameters = dict(locals())
        #with tfp.util.deferred_dependencies.defer_dependencies():
        self._loc = jnp.zeros_like(scale_diag) if loc is None else jnp.asarray(loc)
        self._scale_diag = jnp.asarray(scale_diag)
        
        self.beta = beta
        self.gen = ColoredNoiseProcess(beta=self.beta, size=(scale_diag.shape[-1], seq_len), key=key)

        if self._loc.shape != self._scale_diag.shape:
            raise ValueError(f"Shape mismatch: loc {self._loc.shape} vs scale_diag {self._scale_diag.shape}")
        
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])
        self._event_shape_ = self._loc.shape[-1:]
        
        super().__init__(
            dtype=self._loc.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @property
    def loc(self):
        return self._loc

    @property
    def scale_diag(self):
        return self._scale_diag

    def _batch_shape(self):
        return self._batch_shape_

    def _event_shape(self):
        return self._event_shape_

    def _sample_n(self, sample_shape, seed):
        key = jax.random.split(seed)[0]
        if type(sample_shape) is not tuple:
            sample_shape = (sample_shape,)
        if self._loc.shape[0] == 1:
            eps = jnp.array(self.gen.sample())[None]
        else:
            eps = jax.random.normal(key, shape=sample_shape + self._batch_shape_ + self._event_shape_)
        return self._loc + eps * self._scale_diag

    def _log_prob(self, value):
        var = jnp.square(self._scale_diag)
        log_scale = jnp.log(self._scale_diag)
        return -0.5 * (jnp.square(value - self._loc) / var + 2. * log_scale + jnp.log(2. * jnp.pi)).sum(axis=-1)

    def _mean(self):
        return self._loc

    def _stddev(self):
        return self._scale_diag

    def set_mean_and_scale_diag(self, mean, scale_diag):
        self._loc = mean
        self._scale_diag = scale_diag
        self._batch_shape_ = jax.lax.broadcast_shapes(self._loc.shape[:-1], self._scale_diag.shape[:-1])

    def _entropy(self):
        return jnp.sum(
            jnp.log(self._scale_diag * jnp.sqrt(2. * jnp.pi * jnp.e)),
            axis=-1
        )

    def mode(self):
        return self._loc

    def __repr__(self) -> str:
        return f"ColoredNoiseDist(beta={self.beta})"