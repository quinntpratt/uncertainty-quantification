#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:22:18 2021

@author: Quinn Pratt

Investigation of the distribution of Y = X^2 when X is a normally-
distributed random variable.

Presented at Plasmapy Hack-Week 2021
This script was used to produce the plots on slide 6.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from math import sqrt, pi, gamma
from uncertainties import ufloat

def gaussian(x, mu, sigma):
    ''' function returning the pdf of a guassian-distributed variable.
    :param x: np.array-type variable over which to evaluate the pdf.
    :param mu: mean
    :param sigma: std.dev
    '''
    return 1/sqrt(2*pi*sigma**2)*np.exp(-(x - mu)**2/(2*sigma**2))

def y_pdf_analytic(y, mu, sigma):
    ''' function returning the distribution of the random variable y = x^2
    where x ~ N(mu, sigma^2).
    Compare with wiki, 
    https://en.wikipedia.org/wiki/Random_variable#Example_4
    :param y: np.array-type of y's over which to evaluate the PDF.
    :param mu: mean of the original x-pdf.
    :param sigma: std.dev of the original x-pdf. 
    '''
    prefactor = 1/sqrt(2*pi)/sigma * 1/(2*np.sqrt(y))
    term_1 = np.exp(-(np.sqrt(y) - mu)**2/(2*sigma**2))
    term_2 = np.exp(-(np.sqrt(y) + mu)**2/(2*sigma**2))
    return prefactor*(term_1 + term_2)

def non_central_chi2(z, k, lam):
    ''' function returning the pdf of the non-central chi-squared distribution
    Compare with wiki,
    https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
    :arg z: random variable to eval. the pdf over.
    :arg k: DOF
    :arg lam: non-centrality param.
    '''
    term_1 = 0.5*np.exp(-0.5*(z + lam))
    term_2 = (z/lam)**(0.25*k - 0.5)
    term_3 = sps.iv(0.5*k - 1, np.sqrt(lam*z))
    return term_1 * term_2 * term_3

def y_moments_analytic(mu, sigma, k):
    ''' function returning the k-th moment of the random variable, 
        y = x^2 where x ~ N(mu, sigma^2).
    :arg mu: mean of the underlying x distr
    :arg sigma: std.dev of the underlying x distr.
    :arg k: integer moment desired.
    '''
    prefactor = 1/sqrt(pi) * 2**k * sigma**(2*k) * gamma(k + 0.5)
    term_1 = sps.hyp1f1(-k, 0.5, -mu**2/(2*sigma**2))
    return prefactor * term_1
    

# 1. Control Params,
mu = 2.0 # mean
sigma = 0.1 # std.dev

# 2. Sampling,
Ns = 10000 # No. of random samples to generate from gaussian
Xs = np.random.normal(mu, sigma, size=Ns) 
Ys = np.square(Xs) # squaring produces *new* samples from the y distr.

# 3. Numerical Integration,
Ni = 1024 # No. of points in the y grid for evaluations,
yi = np.linspace(1E-4, 20, Ni) # array of y's over which to evaluate the pdf.

# 4. Plot,
fig = plt.figure(num="Y = X^2 Distribution")
# - plot the analytic form of the PDF,
plt.plot(yi, y_pdf_analytic(yi, mu, sigma),'b', lw=2, label='Analytic')
# - plot a histogram of the sampled results, 
plt.hist(Ys, bins=20, density=True, 
         edgecolor='k', color='c', alpha=0.5, 
         label=f'Sampling, N = {Ns}')

# 5. Bonus: check wiki claim relating this distribution to the 1-dof chi-sq.
# On wiki they say that the analytic form is *equivalent* to a non-central 
# chi-squared distribution with 1 dof... this is not exactly true, one must 
# perform some scalings to get things to work out...

z = yi/sigma**2 # scale y's by the variance, 
# the NCX2 distribution is also scaled by the variance,
fy = non_central_chi2(z, 1, mu**2/sigma**2)/sigma**2 
# uncomment this line to compare with the analytic result,
#plt.plot(yi, fy, 'm-')

# 6. Linear error propagation theory,
y_mu_approx = mu**2
y_sigma_approx = sqrt(4*sigma**2*mu**2)
# - extended grid to evaluate the approximate distribution (pretend it's gaussian)
yi_ext = np.linspace(-5, 20, Ni)
fapprox = gaussian(yi_ext, y_mu_approx, y_sigma_approx)
plt.plot(yi_ext, fapprox, 'r-', lw=2, label='Linear Theory')

# 7. Plot Formatting, 
fs = 20
plt.xlim([-5, 25])
plt.xlim([1, 7])
plt.ylim([.0, 0.2])
plt.ylim([.0, 1.25])
plt.legend(loc='upper right',fontsize=fs-8)
plt.xlabel('y',fontsize=fs-2)
plt.ylabel('density')
plt.title(r'$x = {0:.1f} \pm {1:.1f}$'.format(mu, sigma),fontsize=fs)
plt.setp(plt.gca().get_xticklabels(),fontsize=fs-2)
fig.tight_layout()
# uncomment to save the figure, 
#plt.savefig('demo_x2.png', transparent=False, dpi=200)


# 8. The 1st Moment (average)
# - there are various ways to compute mu_y i.e. <y>
samp_avg = np.average(Ys) # from sampling
# vs. numerically integrating the analytic PDF,
int_avg = np.trapz(yi*y_pdf_analytic(yi, mu, sigma), yi)
# vs. the analytic result,
exact_avg = y_moments_analytic(mu, sigma, 1)
# vs. linear error propagation theory,
lin_avg = y_mu_approx

print(' --- Mean of Y --- ')
print(f'* Sampling E1 = {samp_avg}')
print(f'* Trapz E1 = {int_avg}')
print(f'* Analytic E1 = {exact_avg}')
print(f'* Linear Error Propagation E1 = {lin_avg}')

# 9. The Variance,
samp_var = np.var(Ys) # from sampling
# vs. numerically integrating,
int_var = np.trapz(yi**2*y_pdf_analytic(yi, mu, sigma), yi) - int_avg**2
# vs. the analytic result,
exact_var = y_moments_analytic(mu, sigma, 2) - exact_avg**2
# vs. linear error propagation...
lin_var = y_sigma_approx**2

print(' --- Variance of Y --- ')
print(f'* Sampling Var = {samp_var}')
print(f'* Trapz Var = {int_var}')
print(f'* Analytic Var = {exact_var}')
print(f'* Linear Error Propagation Var = {lin_var}')

# 10. Compare with the uncertainties package,
x = ufloat(mu, sigma)
y = x**2
print(' --- Uncertainties --- ')
print(y)
