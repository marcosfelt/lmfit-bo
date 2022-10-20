# LMfit-BO

## Overview

LMFit-BO provides a Least-Squares Minmization class for fitting data to models that are expensive to evaluate.  It replaces the scipy-optimize methods in the original [LMfit-py](https://github.com/lmfit/lmfit-py) with Bayesian optimization.

## LMfit-BO vs LMFit-py

The original LMfit-py should be the default choice for fitting data to models that are "cheap" to evaluate. Practically, cheap means a single model evaluation for a set of parameters takes on the order of milliseconds to seconds.

However, there are many cases where the model being fitted takes several minutes to hours to evaluate for each set of parameters. These expensive-to-evaluate models are common in a wide range of fields including chemistry, biology, materials, and engineering.  LMfit-BO is ideal for such cases.

## Parameters and Fitting

LMfit-BO uses the same easy-to-use API as LMfit-py. 

You can specify parameters using dictionary syntax:
```python
fit_params = Parameters()
fit_params['amp'] = Parameter(value=1.2, min=0.1, max=1000)
fit_params['cen'] = Parameter(value=40.0, vary=False)
fit_params['wid'] = Parameter(value=4, min=0)
```

or via function methods:
```python
fit_params = Parameters()
fit_params.add('amp', value=1.2, min=0.1, max=1000)
fit_params.add('cen', value=40.0, vary=False)
fit_params.add('wid', value=4, min=0)
```

Then, you write a function to be minimized (in the least squares sense) with the first argument being the `Parameters` object and additional positional and keyword arguments as desired:

```python
def myfunc(params, x, data, someflag=True):
    amp = params['amp'].value
    cen = params['cen'].value
    wid = params['wid'].value
    ...
    return residual_array
```
The function should return the residual (i.e., data-model) array to be minimized.

To perform fitting, the user calls:

```python
result = minimize(myfunc, fit_params, args=(x, data), kws={'someflag':True}, ....)
```

