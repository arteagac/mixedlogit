"""
Implements multinomial and conditional logit models
"""
# pylint: disable=line-too-long,invalid-name

import numpy as np

class MultinomialLogit():
    """Class for estimation of Multinomial and Conditional Logit Models"""

    def fit(self, X, y, initial_coeff=None, maxiter=2000):
        if initial_coeff:
            betas = initial_coeff
            if len(initial_coeff) != len(X):
                raise ValueError("The size of initial_coeff must be: "
                                 + int(len(X)))
        else:
            betas = np.repeat(.0, X.shape[2])
            
        y = y.reshape(X.shape[0], X.shape[1])

        # Call optimization routine
        optimize_res = self._bfgs_optimization(betas, X, y, maxiter)
        return optimize_res

    def _compute_probabilities(self, betas, X):
        XB = X.dot(betas)
        eXB = np.exp(XB)
        p = eXB/np.sum(eXB, axis=1, keepdims=True)
        return p
    
    def _loglik_and_gradient(self, betas, X, y):
        """
        Computes the log likelihood of the parameters B with self.X and self.y data

        Parameters
        ----------
                B: numpy array, shape [n_parameters, ], Vector of betas or parameters

        Returns
        ----------
                ll: float, Optimal value of log likelihood (negative)
                g: numpy array, shape[n_parameters], Vector of individual gradients (negative)
                Hinv: numpy array, shape [n_parameters, n_parameters] Inverse of the approx. hessian
        """
        p = self._compute_probabilities(betas, X)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.sum(np.log(lik))
        # Gradient
        g_i = np.einsum('nj,njk -> nk', (y-p), X)  # Individual contribution to the gradient

        H = np.dot(g_i.T, g_i)
        Hinv = np.linalg.inv(H)
        grad = np.sum(g_i, axis=0)
        return (-loglik, -grad, Hinv)

    def _bfgs_optimization(self, betas, X, y, maxiter):
        """
        Performs the BFGS optimization routine. For more information in this newton-based
        optimization technique see: http://aria42.com/blog/2014/12/understanding-lbfgs

        Parameters
        ----------
                betas: numpy array, shape [n_parameters, ], Vector of betas or parameters

        Returns
        ----------
                B: numpy array, shape [n_parameters, ], Optimized parameters
                res: float, Optimal value of optimization function (log likelihood in this case)
                g: numpy array, shape[n_parameters], Vector of individual gradients (negative)
                Hinv: numpy array, shape [n_parameters, n_parameters] Inverse of the approx. hessian
                convergence: bool, True when optimization converges
                current_iteration: int, Iteration when convergence was reached
        """
        res, g, Hinv = self._loglik_and_gradient(betas, X, y)
        current_iteration = 0
        convergence = False
        while True:
            old_g = g

            d = -Hinv.dot(g)

            step = 2
            while True:
                step = step/2
                s = step*d
                betas = betas + s
                resnew, gnew, _ = self._loglik_and_gradient(betas, X, y)
                if resnew <= res or step < 1e-10:
                    break

            old_res = res
            res = resnew
            g = gnew
            delta_g = g - old_g

            Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
                delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                    Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                    (s.dot(delta_g)))
            current_iteration = current_iteration + 1
            if np.abs(res - old_res) < 0.00001:
                convergence = True
                break
            if current_iteration > maxiter:
                convergence = False
                break

        return {'success':convergence, 'x': betas, 'fun': res, 'hess_inv': Hinv,
                'nit': current_iteration}
