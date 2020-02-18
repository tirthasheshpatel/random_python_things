import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


def make_dataset_using_clt(rng, n_samples):
    r"""Generate Gaussian Samples in a given range.
    Uses Central limit theorem.

    Parameters
    -----
    rng : float, >0
          range of the samples ``(0, rng)``

    n_samples : int, >0
                number of samples to generate

    Returns
    -----
    norm_samples : array_like
                   Gaussian Samples

    ======      =======================
    Mean        .. :math: \mu &= rng / 2
    Varience    .. :math: \sigma^2 &= rng / 12
    ======      =======================
    """
    if n_samples < 0:
        raise ValueError("n_samples must be > 0")

    #  generate from uniform
    unif_samples = st.uniform(0.0, 1.0).rvs(rng * n_samples).reshape(n_samples, rng)

    # apply central limit theorem
    norm_samples = np.sum(unif_samples, axis=1)

    return norm_samples


def make_dataset_using_truncated_norm(mu, std, rng, n_samples):
    """Generates Gaussian Samples in a given range
    Uses Truncated Normal Distribution.

    Parameters
    -----
    rng : float
          range of the samples

    n_samples : int
                number of samples to generate
    
    Returns
    -----
    norm_samples : array_like
                   Samples from Gaussian Distribution
    """
    a, b = (0 - mu) / std, (rng - mu) / std
    return st.truncnorm(a, b).rvs(n_samples) * std + mu


def evaluate_z(samples):
    return (samples - np.mean(samples)) / np.std(samples)


if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10.5, 7.5)
    samples = make_dataset_using_truncated_norm(50, 10, 100, 200)
    # samples = make_dataset_using_clt(100, 200)
    mu = samples.mean()
    std = np.std(samples)
    z_score = evaluate_z(samples)
    A = z_score >= 2
    B = (-0.5 <= z_score) & (z_score <= 0.5)
    C = (0.5 < z_score) & (z_score <= 2)
    D = (-2.0 <= z_score) & (z_score < -0.5)
    E = z_score < -2
    ax[0].bar(
        x=["A", "B", "C", "D", "E"],
        height=[A.sum(), B.sum(), C.sum(), D.sum(), E.sum()],
        width=1,
    )
    plt.title(
        f"A {samples[A].min():.1f}, {samples[A].max():.1f} "
        f"B {samples[B].min():.1f}, {samples[B].max():.1f} "
        f"C {samples[C].min():.1f}, {samples[C].max():.1f} "
        f"D {samples[D].min():.1f}, {samples[D].max():.1f}\n"
        f"E {samples[E].min():.1f}, {samples[E].max():.1f}\n"
    )
    ax[1].hist(samples, bins=5, density=True)
    plt.show()
