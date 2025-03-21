import numpy as np
import torch
from sbx.noise.colorednoise import powerlaw_psd_gaussian
from scipy.signal import butter, lfilter, periodogram

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution

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
    def __init__(self, cutoff, order, sampling_freq, size, scale=1, rng=None):
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
        self.rng = rng if rng is not None else np.random.default_rng()

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
        self.beta = 1.
        self.minimum_frequency = 0.
        #self.size = (1000, 1000)
        #self.buffer_ = powerlaw_psd_gaussian(
        #        exponent=self.beta, size=self.size, fmin=self.minimum_frequency, rng=self.rng)
        # fill the buffer with the low-pass filtered white noise signal using numpy and scipy
        self.buffer = self.rng.normal(size=self.size)
        self.buffer = lfilter(self.b, self.a, self.buffer)
        #self.buffer = self.buffer / np.std(self.buffer, axis=-1, keepdims=True)
        self.buffer = self.buffer / np.std(self.buffer, axis=-1).mean()

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

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]


class LowPassNoiseDist(SquashedDiagGaussianDistribution):
    def __init__(self, cutoff, order, sampling_freq, seq_len, action_dim=None, rng=None, epsilon=1e-6):
        """
        Gaussian colored noise distribution for using colored action noise with stochastic policies.

        The colored noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        cutoff : float or array_like
            If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        order : int
        sampling_freq : float
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
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        assert (action_dim is not None) == np.isscalar(cutoff), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        if np.isscalar(cutoff):
            super().__init__(action_dim, epsilon)
            self.cutoff = cutoff
            self.order = order
            self.sampling_freq = sampling_freq
            self.gen = LowPassNoiseProcess(cutoff=self.cutoff, order=self.order, sampling_freq=self.sampling_freq, size=(action_dim, seq_len), rng=rng)
            #self.beta = 1.
            #self.gen_ = ColoredNoiseProcess(beta=self.beta, size=(action_dim, seq_len), rng=rng)
        else:
            assert len(cutoff) == action_dim, "Length of `beta` has to match `action_dim`."
            assert len(order) == len(sampling_freq) == len(cutoff), "Length of `order` and `sampling_freq` has to match `cutoff`."
            super().__init__(len(cutoff), epsilon)
            self.cutoff = np.asarray(cutoff)
            self.order = np.asarray(order)
            self.sampling_freq = np.asarray(sampling_freq)
            self.gen = [LowPassNoiseProcess(cutoff=c, order=o, sampling_freq=s, size=seq_len, rng=rng) for c, o, s in zip(self.cutoff, self.order, self.sampling_freq)]

    def sample(self) -> torch.Tensor:
        if np.isscalar(self.cutoff):
            lpn_sample = torch.tensor(self.gen.sample()).float()
        else:
            lpn_sample = torch.tensor([lpnp.sample() for lpnp in self.gen]).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev * lpn_sample
        return torch.tanh(self.gaussian_actions)

    def __repr__(self) -> str:
        return f"LowPassNoiseDist(cutoff={self.cutoff}, order={self.order})"


class LowPassNoiseUnsquashedDist(DiagGaussianDistribution):
    def __init__(self, cutoff, order, sampling_freq, seq_len, action_dim=None, rng=None):
        """
        Gaussian colored noise distribution for using colored action noise with stochastic policies.

        The colored noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`DiagGaussianDistribution`).

        Parameters
        ----------
        cutoff : float or array_like
            If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        order : int
        sampling_freq : float
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
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        assert (action_dim is not None) == np.isscalar(cutoff), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        if np.isscalar(cutoff):
            super().__init__(action_dim)
            self.cutoff = cutoff
            self.order = order
            self.sampling_freq = sampling_freq
            self.gen = LowPassNoiseProcess(cutoff=self.cutoff, order=self.order, sampling_freq=self.sampling_freq, size=(action_dim, seq_len), rng=rng)
            #self.beta = 1.
            #self.gen_ = ColoredNoiseProcess(beta=self.beta, size=(action_dim, seq_len), rng=rng)
        else:
            assert len(cutoff) == action_dim, "Length of `beta` has to match `action_dim`."
            assert len(order) == len(sampling_freq) == len(cutoff), "Length of `order` and `sampling_freq` has to match `cutoff`."
            super().__init__(len(cutoff))
            self.cutoff = np.asarray(cutoff)
            self.order = np.asarray(order)
            self.sampling_freq = np.asarray(sampling_freq)
            self.gen = [LowPassNoiseProcess(cutoff=c, order=o, sampling_freq=s, size=seq_len, rng=rng) for c, o, s in zip(self.cutoff, self.order, self.sampling_freq)]

    def sample(self) -> torch.Tensor:
        if np.isscalar(self.cutoff):
            lpn_sample = torch.tensor(self.gen.sample()).float()
        else:
            lpn_sample = torch.tensor([lpnp.sample() for lpnp in self.gen]).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev * lpn_sample
        return self.gaussian_actions

    def __repr__(self) -> str:
        return f"LowPassNoiseUnsquashedDist(cutoff={self.cutoff}, order={self.order})"