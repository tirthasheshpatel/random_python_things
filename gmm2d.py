import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal as Normal
from sklearn.datasets import make_blobs, make_spd_matrix
np.random.seed(42)

def make_simplex(n_clusters):
    """Generate a (n_clusters - 1)-D random simplex"""
    x = np.random.rand(n_clusters)
    return np.exp(x)/np.sum(np.exp(x))

class GaussianMM:
    """Gaussian Mixture Model. A soft clustering
    method using EM Algorithm.

    Parameters
    -----
    n_clusters: int, optional
                number of gaussian clusters to use.

    Examples
    -----
    >>> from gmm2d import GaussianMM
    >>> model = GaussianMM(n_clusters=2)
    >>> model.fit(X, epochs=100)
    """
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.n_samples = None
        self.n_features = None
        # we may later need these to run the e-step.
        # NOTE: all the class variables MUST be declared
        # here but can be initialized outside!
        self.posterior = None
        self.vlb = -np.inf
        self.mu = None
        self.sigma = None
        self.pi = None
    
    def __init_params(self, n_samples, n_features):
        """Reinitialize the parameters.

        So, here's our parameters:
        ===========================================================================
        Parameters
        -----
        n_samples : number of data points in the dataset used during fitting.
        n_features : dimentions of the dataset used during fitting.
        pi : weights of `n_clusters` gaussians aka a categorical distribution over
             latent variable.
        mu : centers of `n_clusters` gaussians aka mean of the gaussians
        sigma : shape parameter of the `n_clusters` gaussians aka covarience
                matrix of the gaussians.
        posterior : posterior distribution over the latent varables.
        ===========================================================================
        Shapes
        -----
        n_samples : scalar
        n_features : scalar
        pi : Row Vector of shape `(n_clusters, 1)`
        mu : Matrix of shape `(n_clusters, n_features)`
        sigma : Tensor of shape `(n_clusters, n_features, n_features)`
        posterior : Row vector of shape `(n_clusters, n_samples)`
        ===========================================================================
        """
        # initialize the scalars
        self.n_samples = n_samples
        self.n_features = n_features

        # initialize the `theta` parameters
        self.pi = make_simplex(self.n_clusters).reshape(-1, 1)
        self.mu = np.random.rand(self.n_clusters, self.n_features)
        self.sigma = np.asarray([make_spd_matrix(self.n_features) \
                                 for _ in range(self.n_clusters)])
        
        # Initialize the posterior as zeros
        self.posterior = np.zeros((self.n_clusters, self.n_samples), dtype=np.float64)

        return None

    def _e_step(self, X):
        """Calculate the posterior which will
        minimize the gab between VLB and log-likelihood.
        Keywords: Variational Lower Bound (VLB), log-likelihood
        =====================================================
        $$\min _{\theta} \left[ \log(P(X|\theta)) - L(\theta^k, q) \right]$$
        $$L(\theta^k, q) = \sum_{i=0}^{N} \sum_{c=1}^{C} q(t_i=c) \log\left(\frac{P(x_i, t_i=c|\theta^k_c)}{q(t_i=c)}\right)$$

        which is equivalent to minimizing $KL(q(t_i=c) || P(t_i=c/x_i, \theta^k_c))$.

        Hence, we just need to set the latent distribution $q$
        equal to the posterior $P(t_i|x_i, \theta^k_c)$ in the E-Step.
        =====================================================
        Shapes
        -----
        posterior : Row vector of shape `(n_clusters, 1)`
        =====================================================
        """
        # First, calculate the normalization constant of the posterior.
        # NOTE: It is tractable to calculate the porterior as the latent
        # variable is finitely discrete.
        _normalization_constant = 0.
        for cluster in range(self.n_clusters):
            _normalization_constant += Normal(self.mu[cluster], self.sigma[cluster]).pdf(X) \
                                      * self.pi[cluster]
        
        # Now, we calculate the posterior for each cluster.
        for cluster in range(self.n_clusters):
            self.posterior[cluster] = Normal(self.mu[cluster], self.sigma[cluster]).pdf(X) \
                                         * self.pi[cluster] \
                                         / _normalization_constant
        
        return None

    def _m_step(self, X):
        """Maximization Step in EM Algorithm. Computes the
        parameters analytically that maximize the VLB.
        ==================================================
        In M-Step we maximize:
        $$L(\theta, q^k) = \sum _{i=1} ^{N} \sum _{c=1} ^{C} q^k(t_i=c)\log\left(\frac{P(x_i, t_i=c|\theta_c)}{q^k(t_i=c)}\right)$$

        This function can be diffrentiatied w.r.t to our parameters
        to get the following equations:

        Mean : $$\mu_c = \frac{\sum_{i=1}^{N} q(t_i=c)x_i}{\sum_{i=1}^{N} q(t_i=c)}$$

        Varience : $$\Sigma_c = \frac{\sum_{i=1}^{N} q(t_i=c)(x_i-\mu_c)(x_i-\mu_c)^T}{\sum_{i=1}^{N} q(t_i=c)}$$

        Weights : $$\pi_c = \frac{\sum_{i=1}^{N}q(t_i=c)}{N}$$
        ==================================================
        """
        # First, update the centers of clusters
        for cluster in range(self.n_clusters):
            self.mu[cluster] = np.sum(self.posterior[cluster] * X.T, axis=1) \
                               / np.sum(self.posterior[cluster])
        
        for cluster in range(self.n_clusters):
            self.sigma[cluster] = np.sum([self.posterior[cluster, i] \
                                          * np.outer( X[i] - self.mu[cluster], X[i] - self.mu[cluster] ) \
                                          for i in range(self.n_samples)], axis=0) \
                                  / np.sum(self.posterior[cluster])
        
        self.pi = np.sum(self.posterior, axis=1) \
                           / self.n_samples
        
        return None

    def _step(self, X):
        """Take one step of EM Algorithm"""
        # E-Step : Calculate the posterior with thetas fixed
        self._e_step(X)
        # M-Step : Calculate the values of new parameters by
        # differentiating the VLB.
        self._m_step(X)

    def _init_animate(self):
        """Initialize the animation"""
        self.ax.clear()
        self.__init_params(*self.X.shape)

        labels = self.posterior.argmax(axis=0)
        colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.

        zm = self.pi[0]*Normal.pdf(self.grid, mean=self.mu[0], cov=self.sigma[0])
        for cluster in range(1, self.n_clusters):
            zm += self.pi[cluster]*Normal.pdf(self.grid, mean=self.mu[cluster], cov=self.sigma[cluster])

        self.cf = self.ax.contourf(self.xm, self.ym, zm, alpha=0.6)
        self.sc = self.ax.scatter(self.X[:, 0], self.X[:, 1], c=colors[labels], s=30)
        return self.cf, self.sc

    def _fit_animate(self, frame):
        """Runs an animation while training!"""
        # This is the optional part used for plotting
        # so I can play animations. Not needed to be
        # tested! Enjoy!
        # Restart the animation every 25 steps
        if ((frame+1) % 25) == 0:
            self.__init_params(*self.X.shape)
        self.ax.clear()
        try:
            if np.abs(1. - np.sum(self.pi)) > 1e-9:
                plt.title("Model Collapsed!")
                return self.cf, self.sc
            self._step(self.X)
            vlb = self._vlb(self.X)
            if vlb > self.vlb:
                self.vlb = vlb
        except np.linalg.LinAlgError:
            self.__init_params(*self.X.shape)
            plt.title("Model Collapsed!")
            return self.cf, self.sc

        labels = self.posterior.argmax(axis=0)
        colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.

        zm = self.pi[0]*Normal.pdf(self.grid, mean=self.mu[0], cov=self.sigma[0])
        for cluster in range(1, self.n_clusters):
            zm += self.pi[cluster]*Normal.pdf(self.grid, mean=self.mu[cluster], cov=self.sigma[cluster])

        self.cf = self.ax.contourf(self.xm, self.ym, zm, alpha=0.6)
        self.sc = self.ax.scatter(self.X[:, 0], self.X[:, 1], c=colors[labels], s=30)
        plt.title(f"loss : {vlb:.2f}, best_loss : {self.vlb:.2f}")
        return self.cf, self.sc

    def set_data(self, X):
        """Used only to run animations on particulare datasets."""
        self.X = X

    def _log_likelihood(self, X):
        raise NotImplementedError("I am Lazy!")

    def _vlb(self, X):
        """Evaluates the Variational Lower Bound
        for current values of parameters.
        =========================================
        $$L(\theta, q) = \sum _{i=0} ^{N} \sum _{c=1} ^{C} q(t_i=c) \log \left( P(x_i, t_i=c|\theta_c) \right)$$

        Our objective is to maximize this lower bound. The value of
        VLB is logged on the terminal to make sure that it is increasing.
        NOTE: If the value decreases, then there is something wrong with
        the algorithm. It can be proved that this value should always increase
        after each step.
        =========================================
        Shapes
        -----
        vlb : scalar
        =========================================
        """
        vlb = np.sum(
            [
                self.posterior[cluster] * np.log(self.pi[cluster]) \
                + Normal(self.mu[cluster], self.sigma[cluster]).logpdf(X) \
                for cluster in range(self.n_clusters)
            ]
        ) - np.sum(self.posterior*np.log(self.posterior))

        return vlb

    def fit(self, X, max_iter=100, n_repeats=100, rtol=1e-9, log_vbl=False):
        """Fit a dataset to the model.

        Parameters
        -----
        X : array_like of shape (n_samples, n_features)
            dataset
        
        max_iter : int, optional
                   maximum iterations of the EM-Algorithm.
        
        n_repeats : int, optional
                    The number of times to repeat the algorithm
                    for better convergence.

        rtol : float, optional
               tolerance for divergence of weights of gaussians.
        """
        # start training
        vlb = -np.inf
        best_mu = None
        best_sigma = None
        best_pi = None
        for _ in range(n_repeats):
            # NOTE: we are performing high dimentional
            # optimization which can often lead to divergence
            # of our constraints. In such a case, we simply
            # restart the training.
            try:
                # Initialize the parameters
                self.__init_params(*X.shape)
                # Take `max_iter` steps.
                for __ in range(max_iter):
                    # If the constraint over weights diverge
                    # restart training.
                    if np.abs(1. - np.sum(self.pi)) > rtol:
                        warnings.warn("Weights Diverged!", RuntimeWarning)
                        break
                    self._step(X)
                    vlb = self._vlb(X)
                    if log_vbl:
                        sys.stdout.write(
                            f"\rTrial {_+1}/{n_repeats}, Epoch {__+1}/{max_iter} : "
                            f"loss : {vlb:.2f}, best_loss : {self.vlb:.2f}"
                        )
                
                # We chosse the best parameters
                # that maximize our lower bound
                if vlb > self.vlb:
                    self.vlb = vlb
                    best_mu = self.mu
                    best_sigma = self.sigma
                    best_pi = self.pi
            except np.linalg.LinAlgError:
                warnings.warn(f"Trial {_+1}: Singular Matrix: Components Collapsed", RuntimeWarning)
                continue
        
        # update the parameters according
        # to the best found so far.
        self.mu = best_mu
        self.sigma = best_sigma
        self.pi = best_pi

        return self.vlb, self.mu, self.sigma, self.pi

    def fit_animate(self, X, render_as_mp4=True):
        """Visualize the training of GMMs by
        running an animation in real time!!!!"""
        self.set_data(X)
        self.fig, self.ax = plt.subplots()
        xs = np.linspace(-15., 15., num=500)
        ys = np.linspace(-15., 15., num=500)
        self.xm, self.ym = np.meshgrid(xs, ys)
        self.grid = np.empty((500, 500, 2))
        self.grid[:, :, 0] = self.xm
        self.grid[:, :, 1] = self.ym
        anim = FuncAnimation(self.fig, self._fit_animate, init_func=self._init_animate, frames=100, interval=50, blit=False)
        if render_as_mp4:
            anim.save('gmm2d_animation.mp4')
        else:
            plt.show()

        return None

    def predict(self, X):
        raise NotImplementedError("I am lazy!")

    def get_params(self):
        """Get a copy of parameters of the model
        which is a tuple (mu, sigma, pi)."""
        return self.mu.copy(), self.sigma.copy(), self.pi.copy()

if __name__ == '__main__':
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=2.)
    model = GaussianMM(n_clusters=3)
    # Please work! Please!!!
    model.fit_animate(X)
