"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
import scipy.stats
from scipy.optimize import minimize

import numpy as np
xnp = np  # xnp is dinamycally linked to numpy or cupy. Numpy used by default
use_gpu = False
try:
    import cupy as cp
    xnp = cp
    use_gpu = True
except ImportError:
    pass


class MixedLogit():
    """Class for estimation of Mixed Logit Models"""

    def __init__(self):
        """Init Function"""
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, rvpos, rvdist, mixby=None, init_coeff=None,
            n_draws=200, halton=True, maxiter=2000):
        N, J, K, R = X.shape[0], X.shape[1], X.shape[2], n_draws

        if mixby is not None:  # If panel
            P = np.max(np.unique(
                mixby, return_counts=True)[1]/J).astype(int)  # Panel size
            N = int(N/P)
        else:
            P = 1

        X = X.reshape(N, P, J, K)
        y = y.reshape(N, P, J, 1)

        # Variable that contains a boolean index of random variables.
        self.rvidx = np.zeros(K, dtype=bool)
        self.rvidx[rvpos] = True  # True: Random var, False: Fixed var
        self.rvdist = rvdist

        draws = self._generate_draws(N, R, halton)  # (N,Kr,R)
        n_coeff = K + len(rvpos)
        if init_coeff is None:
            betas = np.repeat(.0, K + len(rvpos))
        else:
            betas = init_coeff
            if len(init_coeff) != n_coeff:
                raise ValueError("The size of init_coeff must be: " + n_coeff)

        if use_gpu:
            X = cp.asarray(X)
            y = cp.asarray(y)
            draws = cp.asarray(draws)
            print("**** GPU Processing Enabled ****")

        optimize_res = minimize(self._loglik_and_gradient, betas, jac=True,
                                args=(X, y, draws), method='BFGS', tol=1e-6,
                                options={'gtol': 1e-6, 'maxiter': maxiter})

        return optimize_res

    def _compute_probabilities(self, betas, X, draws):
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        Xr = X[:, :, :, self.rvidx]   # Data for random coefficients

        XBf = xnp.einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
        XBr = xnp.einsum('npjk,nkr -> npjr', Xr, Br)  # (N,P,J,R)
        V = XBf[:, :, :, None] + XBr  # (N,P,J,R)
        V[V > 700] = 700
        eV = xnp.exp(V)
        # eV[xnp.isposinf(eV)] = 1e+30
        # eV[xnp.isneginf(eV)] = 1e-30
        sumeV = xnp.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV  # (N,P,J,R)
        return p

    def _loglik_and_gradient(self, betas, X, y, draws):
        if use_gpu:
            betas = cp.asarray(betas)
        p = self._compute_probabilities(betas, X, draws)
        # Probability of chosen alternatives
        pch = xnp.sum(y*p, axis=2)  # (N,P,R)
        pch = pch.prod(axis=1)  # (N,R)
        pch[pch == 0] = 1e-30

        Xf = X[:, :, :, ~self.rvidx]
        Xr = X[:, :, :, self.rvidx]

        # Gradient
        ymp = y - p  # (N,P,J,R)
        # For fixed params
        gf = xnp.einsum('npjr,npjk -> nkr', ymp, Xf)
        # For random params
        der = self._compute_derivatives(betas, draws)
        gr_b = xnp.einsum('npjr,npjk -> nkr', ymp, Xr)*der
        gr_w = xnp.einsum('npjr,npjk -> nkr', ymp, Xr)*der*draws
        # Aggregate gradient and divide by scaled probability
        g = xnp.concatenate((gf, gr_b, gr_w), axis=1)  # (N,K,R)
        g = (g*pch[:, None, :])/xnp.mean(pch, axis=1)[:, None, None]  # (N,K,R)
        g = xnp.mean(g, axis=2)  # (N,K)
        grad = xnp.sum(g, axis=0)  # (K,)

        # Log-likelihood
        lik = xnp.mean(pch, axis=1)  # (N,R)
        loglik = xnp.sum(xnp.log(lik))  # (N,)
        if use_gpu:
            grad, loglik = cp.asnumpy(grad), cp.asnumpy(loglik)
        return -loglik, -grad

    def _apply_distribution(self, betas_random, draws):
        for k, dist in enumerate(self.rvdist):
            if dist == 'ln':
                betas_random[:, k, :] = xnp.exp(betas_random[:, k, :])
        return betas_random

    def _compute_derivatives(self, betas, draws):
        _, betas_random = self._transform_betas(betas, draws)
        Kr = self.rvidx.sum()  # Number of random coeff
        N, R = draws.shape[0], draws.shape[2]
        der = xnp.ones((N, Kr, R))
        for k, dist in enumerate(self.rvdist):
            if dist == 'ln':
                der[:, k, :] = betas_random[:, k, :]
        return der

    def _transform_betas(self, betas, draws):
        # Extract coeffiecients from betas array
        Kr = self.rvidx.sum()   # Number of random coeff
        Kf = len(betas) - 2*Kr  # Number of fixed coeff
        betas_fixed = betas[0:Kf]  # First Kf positions
        br_mean, br_sd = betas[Kf:Kf+Kr], betas[Kf+Kr:]  # Remaining positions
        # Compute: betas = mean + sd*draws
        betas_random = br_mean[None, :, None] + draws*br_sd[None, :, None]
        betas_random = self._apply_distribution(betas_random, draws)
        return betas_fixed, betas_random

    def _generate_draws(self, sample_size, n_draws, halton=True):
        if halton:
            draws_function = self._get_halton_draws
        else:
            draws_function = self._get_uniform_draws
        draws = np.stack([
            scipy.stats.norm.ppf(draws_function(sample_size, n_draws))
            for dis in self.rvdist
            ], axis=1)
        return draws

    def _get_uniform_draws(self, sample_size, n_draws):
        return np.random.uniform(0, 1, size=(sample_size, n_draws))

    def _get_halton_draws(self, sample_size, n_draws, symmetric=False, base=2,
                          skip=0,  shuffled=True):
        numbers = []
        skipped = 0
        for i in range(n_draws * sample_size + 1 + skip):
            n, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n += remainder / denom
            if skipped < skip:
                skipped += 1
            else:
                numbers.append(n)

        numbers = np.array(numbers[1:])
        if shuffled:
            np.random.shuffle(numbers)
        if symmetric:
            numbers = 2.0 * numbers - 1.0

        return numbers.reshape(sample_size, n_draws)
