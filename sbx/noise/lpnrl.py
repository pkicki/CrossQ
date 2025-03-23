import jax
import numpy as np
from scipy.signal import butter, lfilter

import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

from jax import numpy as jnp

class LowPassNoiseProcess():
    """Infinite low-pass noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the low-pass noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, cutoff, order, sampling_freq, size, scale=1, key=None):
        """Infinite low-pass noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. 

        Parameters
        ----------
        cutoff : float
        order : int
        sampling_freq : float
        size : int or tuple of int
            Shape of the sampled colored noise signals. The last dimension (`size[-1]`) specifies the time range, and
            is thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled colored noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        self.cutoff = cutoff
        self.order = order
        self.sampling_freq = sampling_freq
        self.b, self.a = butter(self.order, self.cutoff, fs=self.sampling_freq)

        self.scale = scale
        self.key = key

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[-1]

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        """Reset the buffer with a new time series."""

        self.buffer = jax.random.normal(shape=self.size, key=self.key)
        self.buffer = np.array(self.buffer)
        self.buffer = lfilter(self.b, self.a, self.buffer)
        self.buffer = jnp.array(self.buffer)
        #self.buffer = self.buffer / np.std(self.buffer, axis=-1, keepdims=True)
        self.buffer = self.buffer / jnp.std(self.buffer, axis=-1).mean()

        #import matplotlib.pyplot as plt
        ## compute and plot periodograms for the buffers
        #fs, Pxx = periodogram(self.buffer, fs=self.sampling_freq, axis=-1)
        #fs_, Pxx_ = periodogram(self.buffer_, fs=self.sampling_freq, axis=-1)
        #plt.plot(fs, Pxx.mean(0), label='low-pass filtered white noise')
        #plt.plot(fs_, Pxx_.mean(0), label='pink noise')
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.legend()
        #plt.show()


        #for i in range(9):
        #    plt.subplot(3, 3, i + 1)
        #    plt.plot(self.buffer[i], label='low-pass filtered white noise')
        #    plt.plot(self.buffer_[i], label='pink noise')
        #plt.legend()
        #plt.show()
        
        self.idx = 0

    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * jnp.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]


class LowPassNoiseDist(tfd.Distribution):
    def __init__(self, cutoff=1.0, order=1, sampling_freq=20., seq_len=100., key=None, loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, name="MultivariateNormalDiag"):
        parameters = dict(locals())
        #with tfp.util.deferred_dependencies.defer_dependencies():
        self._loc = jnp.zeros_like(scale_diag) if loc is None else jnp.asarray(loc)
        self._scale_diag = jnp.asarray(scale_diag)

        self.cutoff = cutoff
        self.order = order
        self.sampling_freq = sampling_freq
        self.seq_len = seq_len
        
        self.gen = LowPassNoiseProcess(cutoff=self.cutoff, order=self.order, sampling_freq=self.sampling_freq,
                                       size=(scale_diag.shape[-1], self.seq_len), key=key)

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
        return f"LowPassNoiseDist(cutoff={self.cutoff}, order={self.order}, sampling_freq={self.sampling_freq}, seq_len={self.seq_len})"