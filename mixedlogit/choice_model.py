"""
Implements multinomial and mixed logit models
"""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from scipy.stats import t
from .multinomial_logit import MultinomialLogit
from .mixed_logit import MixedLogit
from enum import Enum

ModelType = Enum('ModelType', ['mixed', 'multinom_condit'])


class ChoiceModel():
    """Class for estimation of Mixed, Multinomial and Conditional Logit Models"""

    def __init__(self):
        """Init Function

        Parameters
        ----------
        random_state: an integer used as seed to generate numpy random numbers
        """
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.model_type = None
        self.model = None

    def fit(self, X, y, alternatives=None, varnames=None, asvars=None,
            randvars=None, mixby=None, n_draws=200, base_alt=None,
            fit_intercept=False, maxiter=2000, random_state=None,
            initial_coeff=None):
        """
        Fits multinomial model using the given data parameters.

        Parameters
        ----------
        X: numpy array, shape [n_samples, n_features], Data matrix in long format.
        y: numpy array, shape [n_samples, ], Vector of choices or discrete output.
        alternatives: list, List of alternatives names or codes.
        asvars: list, List of alternative specific variables
        isvars: list, List of individual specific variables
        base_alt: string, base alternative. When not specified, pymlogit uses the first alternative
                in alternatives vector by default.
        max_iterations: int, Maximum number of optimization iterations
        fit_intercept: bool
        """
        if random_state is not None:
            np.random.seed(random_state)
        self._validate_inputs(X, y, alternatives, varnames, asvars,
                              base_alt, fit_intercept, maxiter)

        self.asvars = [] if not asvars else asvars
        self.isvars = list(set(varnames) - set(asvars))
        self.varnames = varnames
        self.fit_intercept = fit_intercept
        self.n_draws = n_draws
        self.randvars = randvars
        self.alternatives = alternatives
        self.base_alt = alternatives[0] if base_alt is None else base_alt
        self.model_type = ModelType.mixed if randvars \
            else ModelType.multinom_condit

        self.J = len(alternatives)

        # Design matrix
        X, Xnames = self._setup_design_matrix(X)

        # Call computation routines
        if self.model_type is ModelType.mixed:
            rvpos = [varnames.index(i) for i in self.randvars.keys()]
            rvdist = list(self.randvars.values())
            self.model = MixedLogit()
            optimize_res = self.model.fit(X, y, rvpos, rvdist,
                                          mixby, initial_coeff, n_draws,
                                          maxiter)
            # TODO: Update when handling unbalanced panels
            N = int(len(X)/self.J) if mixby is None else len(np.unique(mixby))
            fvpos = list(set(range(len(Xnames))) - set(rvpos))
            coeff_names = np.concatenate((Xnames[fvpos], Xnames[rvpos],
                                          np.char.add("sd.", Xnames[rvpos])))
        else:
            self.model = MultinomialLogit()
            optimize_res = self.model.fit(X, y, initial_coeff, maxiter)
            N = int(len(X)/self.J)
            coeff_names = Xnames

        # Save results
        self.convergence = optimize_res['success']
        self.coeff_ = optimize_res['x']
        self.stderr = np.sqrt(np.diag(optimize_res['hess_inv']))
        self.zvalues = self.coeff_/self.stderr
        self.pvalues = 2*t.pdf(-np.abs(self.zvalues), df=N)  # two tailed test
        self.loglikelihood = -optimize_res['fun']
        self.coeff_names = coeff_names
        total_iter = optimize_res['nit']
        if self.convergence:
            print("Optimization succesfully completed after "+str(total_iter)
                  +" iterations. Use .summary() to see the estimated values")
        else:
            print("**** The optimization did not converge after "
                  +str(total_iter) +" iterations.")
            print("Message: "+optimize_res['message'])

    def _setup_design_matrix(self, X):
        J = len(self.alternatives)
        N = int(len(X)/J)
        isvars = self.isvars.copy()
        asvars = self.asvars.copy()
        varnames = self.varnames.copy()

        if self.fit_intercept:
            isvars.insert(0, '_intercept')
            varnames.insert(0, '_intercept')
            X = np.hstack((np.ones(J*N)[:, None], X))

        ispos = [varnames.index(i) for i in isvars]  # Position of IS vars
        aspos = [varnames.index(i) for i in asvars]  # Position of AS vars

        # Create design matrix
        # For individual specific variables
        if isvars:
            # Create a dummy individual specific variables for the alternatives
            dummy = np.tile(np.eye(J), reps=(N, 1))
            # Remove base alternative
            dummy = np.delete(dummy,
                              self.alternatives.index(self.base_alt), axis=1)
            Xis = X[:, ispos]
            # Multiply dummy representation by the individual specific data
            Xis = np.einsum('ij,ik->ijk', Xis, dummy)
            Xis = Xis.reshape(N, J, (J-1)*len(ispos))

        # For alternative specific variables
        if asvars:
            Xas = X[:, aspos]
            Xas = Xas.reshape(N, J, -1)

        # Set design matrix based on existance of asvars and isvars
        if asvars and isvars:
            X = np.dstack((Xis, Xas))
        elif asvars:
            X = Xas
        elif isvars:
            X = Xis

        names = np.array([isvar+"."+j for isvar in isvars for j in
                          self.alternatives if j != self.base_alt] + asvars)

        return X, names

    def _validate_inputs(self, X, y, alternatives, varnames, asvars, base_alt,
                         fit_intercept, max_iterations):
        if not varnames:
            raise ValueError('The parameter varnames is required')
        if not alternatives:
            raise ValueError('The parameter alternatives is required')
        if X.ndim != 2:
            raise ValueError('X must be an array of two dimensions in long format')
        if y.ndim != 1:
            raise ValueError('y must be an array of one dimension in long format')
        if len(varnames) != X.shape[1]:
            raise ValueError('The length of varnames must match the number of columns in X')

    def summary(self):
        """
        Prints in console the coefficients and additional estimation outputs
        """
        if self.coeff_ is None:
            print('The current model has not been yet estimated')
            return
        if not self.convergence:
            print('***********************************************************')
            print("""WARNING: Convergence was not reached during estimation. 
                  The given estimates may not be reliable""")
            print('***********************************************************')
        print("----------------------------------------------------------------------------------------")
        print("Coefficient          \tEstimate \tStd. Error \tz-value \tP(>|z|)     ")
        print("----------------------------------------------------------------------------------------")
        fmt = "{:16.22} \t{:0.10f} \t{:0.10f} \t{:0.10f} \t{:0.10f} {:5}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 1e-15:
                signif = '***'
            elif self.pvalues[i] < 0.001:
                signif = '**'
            elif self.pvalues[i] < 0.01:
                signif = '*'
            elif self.pvalues[i] < 0.05:
                signif = '.'
            print(fmt.format(self.coeff_names[i][:15], self.coeff_[i], 
                             self.stderr[i], self.zvalues[i], self.pvalues[i], 
                             signif))
        print("----------------------------------------------------------------------------------------")
        print('Significance:  *** 0    ** 0.001    * 0.01    . 0.05')
        print('')
        print('Log-Likelihood= %.3f' % (self.loglikelihood))
