"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
import scipy.stats
from scipy.optimize import minimize
from ._choice_model import ChoiceModel

import numpy as np
xnp = np  # xnp is dinamycally linked to numpy or cupy. Numpy used by default
use_gpu = False
try:
    import cupy as cp
    xnp = cp
    use_gpu = True
except ImportError:
    pass


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models"""

    def __init__(self):
        """Init Function"""
        super(MixedLogit, self).__init__()
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, varnames=None, alternatives=None, asvars=None,
            base_alt=None, fit_intercept=False, init_coeff=None, maxiter=2000,
            random_state=None, randvars=None, mixby=None, n_draws=200,
            halton=True, verbose=1):
        self._validate_inputs(X, y, alternatives, varnames, asvars,
                              base_alt, fit_intercept, maxiter)
        self._pre_fit(alternatives, varnames, asvars, base_alt,
                      fit_intercept, maxiter)

        if random_state is not None:
            np.random.seed(random_state)

        X, Xnames = self._setup_design_matrix(X)
        N, J, K, R = X.shape[0], X.shape[1], X.shape[2], n_draws

        if mixby is not None:  # If panel
            P = np.max(np.unique(
                mixby, return_counts=True)[1]/J).astype(int)  # Panel size
            N = int(N/P)
            # TODO: Update when handling unbalanced panels
        else:
            P = 1

        X = X.reshape(N, P, J, K)
        y = y.reshape(N, P, J, 1)

        # Variable that contains a boolean index of random variables.
        self.n_draws = n_draws
        self.randvars = randvars
        rvpos = [np.where(Xnames == rv)[0][0] for rv in self.randvars.keys()]
        self.rvidx = np.zeros(K, dtype=bool)
        self.rvidx[rvpos] = True  # True: Random var, False: Fixed var
        self.rvdist = list(self.randvars.values())

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
            if verbose > 0:
                print("**** GPU Processing Enabled ****")

        optimizat_res = minimize(self._loglik_gradient_hessian, betas,
                                 jac=True, args=(X, y, draws), method='BFGS',
                                 tol=1e-6,
                                 options={'gtol': 1e-6, 'maxiter': maxiter})
        _, _, hess_inv = \
            self._loglik_gradient_hessian(optimizat_res['x'], X, y, draws,
                                          compute_hess_inv=True)
        optimizat_res['hess_inv'] = hess_inv

        fvpos = list(set(range(len(Xnames))) - set(rvpos))
        coeff_names = np.concatenate((Xnames[fvpos], Xnames[rvpos],
                                      np.char.add("sd.", Xnames[rvpos])))

        self._post_fit(optimizat_res, coeff_names, N, verbose)

    def _compute_probabilities(self, betas, X, draws):
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        Xr = X[:, :, :, self.rvidx]   # Data for random coefficients

        XBf = xnp.einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
        XBr = xnp.einsum('npjk,nkr -> npjr', Xr, Br)  # (N,P,J,R)
        V = XBf[:, :, :, None] + XBr  # (N,P,J,R)
        V[V > 700] = 700
        eV = xnp.exp(V)
        sumeV = xnp.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV  # (N,P,J,R)
        return p

    def _loglik_gradient_hessian(self, betas, X, y, draws,
                                 compute_hess_inv=False):
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
        # Aggregate gradient and multiply by scaled probability
        g = xnp.concatenate((gf, gr_b, gr_w), axis=1)  # (N,K,R)
        g = g*(pch[:, None, :]/xnp.mean(pch, axis=1)[:, None, None])  # (N,K,R)
        g = xnp.mean(g, axis=2)  # (N,K)

        if compute_hess_inv:
            H = g.T.dot(g)
            hess_inv = xnp.linalg.inv(H)
        else:
            hess_inv = None

        grad = xnp.mean(g, axis=0)  # (K,)
        # Log-likelihood
        lik = xnp.mean(pch, axis=1)  # (N,R)
        loglik = xnp.sum(xnp.log(lik))  # (N,)
        if use_gpu:
            grad, loglik = cp.asnumpy(grad), cp.asnumpy(loglik)
            hess_inv = cp.asnumpy(hess_inv)
        return -loglik, -grad, hess_inv

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
            return self._get_halton_draws(sample_size, n_draws,
                                          len(self.rvdist))
        else:
            return self._get_random_normal_draws(sample_size, n_draws,
                                                 len(self.rvdist))

    def _get_random_normal_draws(self, sample_size, n_draws, n_vars):
        draws = [np.random.normal(0, 1, size=(sample_size, n_draws))
                 for _ in range(n_vars)]
        draws = np.stack(draws, axis=1)
        return draws

    def _get_halton_draws(self, sample_size, n_draws, n_vars, shuffled=False):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                  109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                  173, 179, 181, 191, 193, 197, 199]

        def halton_seq(length, prime=3, shuffled=False, drop=100):
            h = np.array([.0])
            t = 0
            while len(h) < length + drop:
                t += 1
                h = np.append(h, np.tile(h, prime-1) +
                              np.repeat(np.arange(1, prime)/prime**t, len(h)))
            seq = h[drop:length+drop]
            if shuffled:
                np.random.shuffle(seq)
            return seq

        draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
                            shuffled=shuffled).reshape(sample_size, n_draws)
                 for i in range(n_vars)]
        draws = np.stack(draws, axis=1)
        draws = scipy.stats.norm.ppf(draws)
        return draws
