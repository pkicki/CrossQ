"""Colored noise generation script
Modified from colorednoise package: https://github.com/felixpatzelt/colorednoise
"""

#from numpy.fft import irfft, rfftfreq
import jax
from jax import numpy as jnp
from jax.numpy.fft import rfftfreq, irfft

def powerlaw_psd_gaussian(exponent, size, fmin=0, key=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    rng : np.random.Generator, optional
        Random number generator (for reproducibility). If not passed, a new
        random number generator is created by calling
        `np.random.default_rng()`.


    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    >>> # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)    # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = jnp.sum(s_scale < fmin)   # Index of the cutoff
    #if ix and ix < len(s_scale):
    #    s_scale[:ix] = s_scale[ix]

    #def update_scale(s_scale, ix):
    #    s_scale = s_scale.at[:ix].set(s_scale[ix])
    #    return s_scale

    #s_scale = jax.lax.cond(
    #    (ix > 0) & (ix < len(s_scale)),
    #    update_scale,
    #    lambda s: s,
    #    s_scale,
    #    ix
    #)

    mask = jnp.arange(len(s_scale)) < ix
    # Broadcast s_scale[ix] to all elements
    fill_value = jnp.full_like(s_scale, s_scale[ix])
    # Use the mask to selectively update elements
    s_scale = jnp.where(mask, fill_value, s_scale)

    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    #w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
    w = w.at[-1].set((1 + (samples % 2)) / 2.)
    sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    if key is None:
        key = jax.random.PRNGKey(0)
    sr = s_scale * jax.random.normal(shape=size, key=key)
    si = s_scale * jax.random.normal(shape=size, key=key)

    # Create masks for special frequency components
    is_even = (samples % 2) == 0

    def fix_even(sr, si):
        # Set imaginary part of Nyquist freq to 0
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * jnp.sqrt(2))
        return sr, si

    # Fix DC component
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * jnp.sqrt(2))

    # Conditionally fix Nyquist frequency for even-length signals
    sr, si = jax.lax.cond(
        is_even,
        fix_even,
        lambda sr, si: (sr, si),
        sr,
        si
    )

    # Combine into complex spectrum
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y
