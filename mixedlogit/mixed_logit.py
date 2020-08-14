"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
import numpy as np
import scipy.stats
from scipy.optimize import minimize


class MixedLogit():
    """Class for estimation of Mixed Logit Models"""

    def __init__(self):
        """Init Function"""
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, rvpos, rvdist, mixby=None, initial_coeff=None,
            n_draws=200, maxiter=2000):
        N = X.shape[0]
        J = X.shape[1]
        K = X.shape[2]
        R = n_draws
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
        draws = np.stack([self._get_normal_std_draws(N, R)
                          for dis in self.rvdist], axis=1)  # (N,Kr,R)

        if initial_coeff:
            betas = initial_coeff
            if len(initial_coeff) != K + len(rvpos):
                raise ValueError("The size of initial_coeff must be: "
                                 + int(K + len(rvpos)))
        else:
            betas = np.repeat(.0, K + len(rvpos))

        optimize_res = minimize(self._loglik_and_gradient, betas, jac=True,
                                args=(X, y, draws), method='BFGS', tol=1e-6,
                                options={'gtol': 1e-6, 'maxiter': maxiter})

        return optimize_res

    def _compute_probabilities(self, betas, X, draws):
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        Xr = X[:, :, :, self.rvidx]   # Data for random coefficients

        XBf = np.einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
        XBr = np.einsum('npjk,nkr -> npjr', Xr, Br)  # (N,P,J,R)
        V = XBf[:, :, :, None] + XBr  # (N,P,J,R)
        V[V > 700] = 700
        eV = np.exp(V)
        eV[np.isposinf(eV)] = 1e+30
        eV[np.isneginf(eV)] = 1e-30
        sumeV = np.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV  # (N,P,J,R)
        return p

    def _loglik_and_gradient(self, betas, X, y, draws):
        p = self._compute_probabilities(betas, X, draws)
        # Probability of chosen alternatives
        pch = np.sum(y*p, axis=2)  # (N,P,R)
        pch = pch.prod(axis=1)  # (N,R)
        pch[pch == 0] = 1e-30

        Xf = X[:, :, :, ~self.rvidx]
        Xr = X[:, :, :, self.rvidx]

        # Gradient
        ymp = y - p  # (N,P,J,R)
        # For fixed params
        gf = np.einsum('npjr,npjk -> nkr', ymp, Xf)
        # For random params
        der = self._compute_derivatives(betas, draws)
        gr_b = np.einsum('npjr,npjk -> nkr', ymp, Xr)*der
        gr_w = np.einsum('npjr,npjk -> nkr', ymp, Xr)*der*draws
        # Aggregate gradient and divide by scaled probability
        g = np.concatenate((gf, gr_b, gr_w), axis=1)  # (N,K,R)
        g = (g*pch[:, None, :])/np.mean(pch, axis=1)[:, None, None]  # (N,K,R)
        g = np.mean(g, axis=2)  # (N,K)
        grad = np.sum(g, axis=0)  # (K,)

        # Log-likelihood
        lik = np.mean(pch, axis=1)  # (N,R)
        loglik = np.sum(np.log(lik))  # (N,)
        return -loglik, -grad

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

    def _get_normal_std_draws(self, sample_size, n_draws, shuffled=False):
        normal_dist = scipy.stats.norm(loc=0.0, scale=1.0)
        draws = normal_dist.rvs(size=(sample_size, n_draws))
        if shuffled:
            np.random.shuffle(draws)
        return draws

    def _apply_distribution(self, betas_random, draws):
        for k, dist in enumerate(self.rvdist):
            if dist == 'ln':
                betas_random[:, k, :] = np.exp(betas_random[:, k, :])
        return betas_random

    def _compute_derivatives(self, betas, draws):
        _, betas_random = self._transform_betas(betas, draws)
        Kr = self.rvidx.sum()  # Number of random coeff
        N, R = draws.shape[0], draws.shape[2]
        der = np.ones((N, Kr, R))
        for k, dist in enumerate(self.rvdist):
            if dist == 'ln':
                der[:, k, :] = betas_random[:, k, :]
        return der
