#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 09:11:45 2021

@author: Quinn Pratt

This script demonstrates the use of the uncertainties package in 
conjunction with fitting routines to propagate fit-errors and covariances. 

We use LMFIT for fitting - but scipy.optimize.curve_fit is also possible.
for more info on LMFIT, 
https://lmfit.github.io/lmfit-py/index.html

This program demonstrates...
    1. generating test-data and package it as an uncertainties-uarray.
    2. performing a simple fit with LMFIT and extract the best-fit values and 
        uncertainties.
    3. constructing an uncertainties-correlated_values object from the LMFIT 
        results and evaluate the best-fit model with correlated values.
    4. computing a numerical derivative using the best-fit (this is contrasted 
        with ignoring the covariance between fit parameters).
    5. propagating uncertainties through an operation not expressible with simple
        mathematical functions provided by the unumpy sub-module. In this program 
        the operation is numerical integration with np.trapz.
"""

import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties.unumpy.core import wrap_array_func
import lmfit as lmf

# %% 1. Generate synthetic data with uncertainties to perform a curve-fit,
np.random.seed(32344323) # for random noise

def myTanh(x, amplitude, center, scale, shift):
    ''' function returning a tanh profile with various params.
    '''
    return amplitude * np.tanh(-1*scale*(x - center)) + shift

# - No. of data points,
N = 10
# - X-values of data points,
x = np.linspace(0, 1, 10)
# - data generated from model,
data = myTanh(x, 1.4, 0.5, 10, 2)
# - multiplicative noise, 
noise = np.random.normal(1, 0.1, N)
data *= noise
# - additive noise,
noise = np.random.uniform(-0.5, 0.5, N)
data += noise
# - synthetic errors (normal)
err = np.random.normal(0.3, 0.075, N)

# Cast the "data" as a uarray,
udata = unp.uarray(data, err)

# Plot our "data",
fig, ax = plt.subplots(1,1,num="Data with Fit")
ax.errorbar(x, unp.nominal_values(udata), yerr=unp.std_devs(udata),
             fmt='b--o',ecolor='k',capsize=4,capthick=2,elinewidth=2,
             label='data')
ax.set_xlabel('x')
ax.set_ylabel('y')

# %% 2. Perform an lmfit curve fit,
# - create the model object, 
model = lmf.Model(myTanh)
params = model.make_params()
# - Constrain params,
params['amplitude'].set(value=1, min=0, max=10)
params['center'].set(value=0.4, min=0, max=1)
params['scale'].set(value=1, min=0.1, max=50)
params['shift'].set(value=0, min=-10, max=10)

# - perform the fit using the supplied data with inverse-varaince weights,
# inverse-variance weights are recommended,
# https://en.wikipedia.org/wiki/Non-linear_least_squares#Theory  
result = model.fit(
    unp.nominal_values(udata),
    x=x,
    params=params,
    weights=1/unp.std_devs(udata)**2
    )
# lmfit has a nice fit-report function,
print(result.fit_report())
# extract the best fit and std.err on each param,
params_best_val = [result.params[k].value for k in result.var_names]
params_std_err = [result.params[k].stderr for k in result.var_names]

# 3. Plot the results,  
# - target grid, 
xi = np.linspace(0, 1, 100)
# - evaluate over the target grid, 
fit = result.eval(x=xi)
# - evaluate the uncertaitny bound over x 
# Note: LMFIT automatically does the *correct* thing w.r.t. uncertainties
dely = result.eval_uncertainty(x=xi)
# - plot,
ax.plot(xi, fit, 'g-', label='Fit')
ax.fill_between(xi, fit-dely, fit + dely, color='g', alpha=0.33, label=r'$1\sigma$ w/ correl.')

# compare with the INCORRECT thing: taking the +/- 1\sigma bounds of params,
fit_p1 = myTanh(xi, *[v + e for v, e in zip(params_best_val, params_std_err)])
fit_m1 = myTanh(xi, *[v - e for v, e in zip(params_best_val, params_std_err)]) 
# plot of the WRONG way,
ax.fill_between(xi,fit_m1,fit_p1,
                color='r', alpha=0.33, label=r'$1\sigma$ no correl.')

ax.legend()

# Incorrectly package the array with its uband as a uarray (ignores correlations)
ufit_no_corr = unp.uarray(fit, dely)

# %% 4. Use the uncertainties package to CORRECTLY package params,
# - prepare a dictionary for the uncertainties-formatted params,
uparams= {}
# - create a correlated-values object from the covariance matrix and the params,
corr_params = unc.correlated_values(params_best_val, result.covar)
# - loop over the model params and place them in the dictionary,  
for vi, var in enumerate(result.var_names):
    uparams[var] = corr_params[vi]

# - evaluate the fit using unumpy function to propagate the uncertainties through the model,
ufit_corr = uparams['amplitude']*unp.tanh(-uparams['scale']*(xi - uparams['center'])) + uparams['shift']


# - printing the "error components" shows the difference between these uarrays,
#   i.e. ufit_corr vs. ufit_no_corr.
#   consider just one point of the proifle,
ind_to_print = 0
# - extract the error-components of the point,
#   this is a dictionary where the key is the uncertainties 'tag'
#   and the value is |sigma|*derivative - which is related to its overall weight
#   in determining the uncertainty at that point - but formally isn't *directly* related.
print('* error components, ')
print(ufit_corr[ind_to_print].error_components())
# - however, if we do the same for the uncorrelated object, we can see that it
#   has no 'memory' of the params it was generated from,
print('* uncorrelated components, ')
print(ufit_no_corr[ind_to_print].error_components())


# %% 5. Derivative of the profile - correlated vs. not,
# - we will compute derivatives using a central-difference method to illustrate the difference,

def myCentralDiff(x,y):
    ''' function to compute a derivative in the central diff. sense
    x-spacing is assumed to be constant.
    boundaries are treated with linear extrapolation
    '''
    h = x[1] - x[0]
    dy0 = y[1] - y[0]
    dyN = y[-1] - y[-2]
    _y = np.concatenate([ 
        [dy0 + y[0]],
        y,
        [dyN + y[-1]],
        ])
    return (np.roll(_y, -1) - np.roll(_y, 1))[1:-1]/(2*h)

deriv_corr = myCentralDiff(xi, ufit_corr)
plt.figure(num="Differentiating uncertain profile")
plt.plot(xi, unp.nominal_values(deriv_corr),'g-', zorder=10)
plt.fill_between(xi,
                unp.nominal_values(deriv_corr) - unp.std_devs(deriv_corr),
                unp.nominal_values(deriv_corr) + unp.std_devs(deriv_corr),
                color='g', alpha=0.33, label=r'$1\sigma$ w/ correl.',zorder=10)

deriv_no_corr = myCentralDiff(xi, ufit_no_corr)
plt.plot(xi, unp.nominal_values(deriv_no_corr),'r-')
plt.fill_between(xi,
                unp.nominal_values(deriv_no_corr) - unp.std_devs(deriv_no_corr),
                unp.nominal_values(deriv_no_corr) + unp.std_devs(deriv_no_corr),
                color='r', alpha=0.33, label=r'$1\sigma$ w/out correl.')
plt.xlabel('x')
plt.ylabel(r'$\frac{dy}{dx}$',rotation=0,fontsize=18)
plt.legend()

# %% 6. Demonstration of wrap_array_func
# - Wrap the numpy trapz function to perform integrals with uncertain arrays,

wrapped_trapz = wrap_array_func(np.trapz)
# - integrate the properly prepared profile to get a ufloat for the integral,
I = wrapped_trapz(ufit_corr,xi)
print(f'* Integral of profile: {I}')

