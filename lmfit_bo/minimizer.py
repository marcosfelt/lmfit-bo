# from .bayes_opt import bayes_opt
from .bayes_opt import gp_minimize
from lmfit.minimizer import (
    Minimizer,
    thisfuncname,
    maxeval_warning,
    SCALAR_METHODS,
    MinimizerException,
    AbortFitException,
    HAS_NUMDIFFTOOLS,
)
import numpy as np
import warnings


class NewMinmizer(Minimizer):
    def minimize(self, method="leastsq", params=None, **kws):
        """Perform the minimization.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use. Valid values are:

            - `'leastsq'`: Levenberg-Marquardt (default)
            - `'least_squares'`: Least-Squares minimization, using Trust
              Region Reflective method
            - `'differential_evolution'`: differential evolution
            - `'brute'`: brute force method
            - `'basinhopping'`: basinhopping
            - `'ampgo'`: Adaptive Memory Programming for Global
              Optimization
            - '`nelder`': Nelder-Mead
            - `'lbfgsb'`: L-BFGS-B
            - `'powell'`: Powell
            - `'cg'`: Conjugate-Gradient
            - `'newton'`: Newton-CG
            - `'cobyla'`: Cobyla
            - `'bfgs'`: BFGS
            - `'tnc'`: Truncated Newton
            - `'trust-ncg'`: Newton-CG trust-region
            - `'trust-exact'`: nearly exact trust-region
            - `'trust-krylov'`: Newton GLTR trust-region
            - `'trust-constr'`: trust-region for constrained optimization
            - `'dogleg'`: Dog-leg trust-region
            - `'slsqp'`: Sequential Linear Squares Programming
            - `'emcee'`: Maximum likelihood via Monte-Carlo Markov Chain
            - `'shgo'`: Simplicial Homology Global Optimization
            - `'dual_annealing'`: Dual Annealing optimization

            In most cases, these methods wrap and use the method with the
            same name from `scipy.optimize`, or use
            `scipy.optimize.minimize` with the same `method` argument.
            Thus `'leastsq'` will use `scipy.optimize.leastsq`, while
            `'powell'` will use `scipy.optimize.minimizer(...,
            method='powell')`.

            For more details on the fitting methods please refer to the
            `SciPy documentation
            <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__.

        params : Parameters, optional
            Parameters of the model to use as starting values.
        **kws : optional
            Additional arguments are passed to the underlying minimization
            method.

        Returns
        -------
        MinimizerResult
            Object containing the optimized parameters and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.

        """
        kwargs = {"params": params}
        kwargs.update(self.kws)
        for maxnfev_alias in ("maxfev", "maxiter"):
            if maxnfev_alias in kws:
                warnings.warn(
                    maxeval_warning.format(maxnfev_alias, thisfuncname()),
                    RuntimeWarning,
                )
                kws.pop(maxnfev_alias)

        kwargs.update(kws)

        user_method = method.lower()
        if user_method.startswith("leasts"):
            function = self.leastsq
        elif user_method.startswith("least_s"):
            function = self.least_squares
        elif user_method == "brute":
            function = self.brute
        elif user_method == "basinhopping":
            function = self.basinhopping
        elif user_method == "ampgo":
            function = self.ampgo
        elif user_method == "emcee":
            function = self.emcee
        elif user_method == "shgo":
            function = self.shgo
        elif user_method == "dual_annealing":
            function = self.dual_annealing
        elif user_method == "bayes_opt":
            function = self.bayes_opt
        else:
            function = self.scalar_minimize
            for key, val in SCALAR_METHODS.items():
                if key.lower().startswith(user_method) or val.lower().startswith(
                    user_method
                ):
                    kwargs["method"] = val
        return function(**kwargs)

    def bayes_opt(self, params=None, max_nfev=None, **kws):
        result = self.prepare_fit(params=params)
        result.method = "bayes_opt"
        self.set_max_nfev(max_nfev, 20 * (result.nvarys + 1))

        dimensions = np.array([(p.min, p.max) for p in self.params.values()])
        bayes_opt_kws = dict(
            dimensions=dimensions,
            initial_point_generator="lhs",
        )

        bayes_opt_kws.update(self.kws)
        bayes_opt_kws.update(kws)

        # x0 = result.init_vals
        result.call_kws = bayes_opt_kws
        try:
            # Put Bayesian optimization code here
            ret = gp_minimize(self.penalty, **bayes_opt_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            # result.message = ret.message
            result.residual = self.__residual(ret.x)
            result.nfev -= 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (
            not result.aborted
            and self.calc_covar
            and HAS_NUMDIFFTOOLS
            and len(result.residual) > len(result.var_names)
        ):
            _covar_ndt = self._calculate_covariance_matrix(ret.x)
            if _covar_ndt is not None:
                result.covar = self._int2ext_cov_x(_covar_ndt, ret.x)
                self._calculate_uncertainties_correlations()

        return result


def minimize_bo(
    fcn,
    params,
    method="leastsq",
    args=None,
    kws=None,
    iter_cb=None,
    scale_covar=True,
    nan_policy="raise",
    reduce_fcn=None,
    calc_covar=True,
    max_nfev=None,
    **fit_kws
):
    """Perform the minimization of the objective function.

    The minimize function takes an objective function to be minimized,
    a dictionary (:class:`~lmfit.parameter.Parameters` ; Parameters) containing
    the model parameters, and several optional arguments including the fitting
    method.

    Parameters
    ----------
    fcn : callable
        Objective function to be minimized. When method is `'leastsq-bo'` or
        '`least_squares`', the objective function should return an array
        of residuals (difference between model and data) to be minimized
        in a least-squares sense. With the scalar methods the objective
        function can either return the residuals array or a single scalar
        value. The function must have the signature::

            fcn(params, *args, **kws)

    params : Parameters
        Contains the Parameters for the model.
    method : str, optional
        Name of the fitting method to use. Valid values are:

        - `'leastsq-bo'`: Bayesian optimization directly on least-squares objective


        In most cases, these methods wrap and use the method of the same
        name from `scipy.optimize`, or use `scipy.optimize.minimize` with
        the same `method` argument. Thus `'leastsq'` will use
        `scipy.optimize.leastsq`, while `'powell'` will use
        `scipy.optimize.minimizer(..., method='powell')`

        For more details on the fitting methods please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__.

    args : tuple, optional
        Positional arguments to pass to `fcn`.
    kws : dict, optional
        Keyword arguments to pass to `fcn`.
    iter_cb : callable, optional
        Function to be called at each fit iteration. This function should
        have the signature::

            iter_cb(params, iter, resid, *args, **kws),

        where `params` will have the current parameter values, `iter` the
        iteration number, `resid` the current residual array, and `*args`
        and `**kws` as passed to the objective function.
    scale_covar : bool, optional
        Whether to automatically scale the covariance matrix (default is
        True).
    nan_policy : {'raise', 'propagate', 'omit'}, optional
        Specifies action if `fcn` (or a Jacobian) returns NaN values. One
        of:

        - `'raise'` : a `ValueError` is raised
        - `'propagate'` : the values returned from `userfcn` are un-altered
        - `'omit'` : non-finite values are filtered

    reduce_fcn : str or callable, optional
        Function to convert a residual array to a scalar value for the
        scalar minimizers. See Notes in `Minimizer`.
    calc_covar : bool, optional
        Whether to calculate the covariance matrix (default is True) for
        solvers other than `'leastsq'` and `'least_squares'`. Requires the
        `numdifftools` package to be installed.
    max_nfev : int or None, optional
        Maximum number of function evaluations (default is None). The
        default value depends on the fitting method.
    **fit_kws : dict, optional
        Options to pass to the minimizer being used.

    Returns
    -------
    MinimizerResult
        Object containing the optimized parameters and several
        goodness-of-fit statistics.


    .. versionchanged:: 0.9.0
       Return value changed to :class:`MinimizerResult`.


    Notes
    -----
    The objective function should return the value to be minimized. For
    the Levenberg-Marquardt algorithm from leastsq(), this returned value
    must be an array, with a length greater than or equal to the number of
    fitting variables in the model. For the other methods, the return
    value can either be a scalar or an array. If an array is returned, the
    sum-of- squares of the array will be sent to the underlying fitting
    method, effectively doing a least-squares optimization of the return
    values.

    A common use for `args` and `kws` would be to pass in other data needed
    to calculate the residual, including such things as the data array,
    dependent variable, uncertainties in the data, and other data structures
    for the model calculation.

    On output, `params` will be unchanged. The best-fit values and, where
    appropriate, estimated uncertainties and correlations, will all be
    contained in the returned :class:`MinimizerResult`. See
    :ref:`fit-results-label` for further details.

    This function is simply a wrapper around :class:`Minimizer` and is
    equivalent to::

        fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                           iter_cb=iter_cb, scale_covar=scale_covar,
                           nan_policy=nan_policy, reduce_fcn=reduce_fcn,
                           calc_covar=calc_covar, **fit_kws)
        fitter.minimize(method=method)

    """
    fitter = NewMinmizer(
        fcn,
        params,
        fcn_args=args,
        fcn_kws=kws,
        iter_cb=iter_cb,
        scale_covar=scale_covar,
        nan_policy=nan_policy,
        reduce_fcn=reduce_fcn,
        calc_covar=calc_covar,
        max_nfev=max_nfev,
        **fit_kws
    )
    return fitter.minimize(method=method)
