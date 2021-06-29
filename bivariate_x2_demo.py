#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:22:39 2021

@author: Quinn Pratt

Investigation of the distribution of Y = (x1 + x2)^2 where x1 and x2 are 
random vars distributed acc to a bivariate normal distribution.

Presented at Plasmapy Hack-Week 2021
This script was used to produce the plots on slide 10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.special as sps
from math import sqrt, pi, gamma
import copy as copy

def gaussian(x, mu, sigma):
    ''' function returning the pdf of a guassian-distributed variable.
    :param x: np.array-type variable over which to evaluate the pdf.
    :param mu: mean
    :param sigma: std.dev
    '''
    return 1/sqrt(2*pi*sigma**2)*np.exp(-(x - mu)**2/(2*sigma**2))

def multivariate_normal(x, mu, covar):
    ''' function to compute the n-dimensional multivariate normal given
    a vector of mu's and the covar-matrix.
    see also: scipy.stats.multivariate_normal
    (reproduced here for tutorial purposes)
    cf. https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case
    :arg x: n-by-M set of M points in n-dims over which to evaluate
    :arg mu: length n vector of centers
    :arg covar: [n x n] covariance matrix
    '''
    n, M = np.shape(x)
    z = np.zeros(M)
    for i in range(M):
        z[i] = (x[:,i] - mu).T @ (np.linalg.inv(covar) @ (x[:,i] - mu))
    prefactor = (2*pi*np.linalg.det(covar))**(-1/2)
    return prefactor*np.exp(-0.5*z)

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
# - we define two means, two std.devs, and a correlation coeff. (rho)
mu1, mu2 = 2.0, 3.0
sigma1, sigma2 = 1., 1.5 
rho = -0.75

# stack the params for convinience, 
mus = np.array([mu1, mu2])
sigs = np.array([sigma1, sigma2])
# the covariance matrix, 
covar = np.array( [[sigs[0]**2, rho*sigs[0]*sigs[1]],
                   [rho*sigs[0]*sigs[1], sigs[1]**2]])


# 2. Sampling,
Ns = 10000 # No. of random samples to generate from bivariate normal.
Xs = np.random.multivariate_normal(mus, covar, size=Ns) # returns a (Ns x 2)
# - To compute Y we perform, 
Ys = np.square(np.sum(Xs, axis=1)) # samples from the y distr.
# - compute the mean and std.dev,
y_mu_samp = np.average(Ys)
y_sigma_samp = np.std(Ys)

# - Plot the result,
plt.figure(num="(x1 + x2)^2")
plt.hist(Ys, bins=20, density=True, edgecolor='k', color='c', alpha=0.5, label=f'Sampling, N = {Ns}')

# 3. Analytic,
# - summing two normally distributed (and correlated) random vars produces 
# a new random var... also normally distributed, 
# cf, https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables#Correlated_random_variables

# - these become the params for recycling our result from univariate y = x^2,

mu_sum = sum(mus) # the mean is the sum of the means..
var_sum = sigma1**2 + sigma2**2 + 2*rho*sigma1*sigma2 # variance is different

Ni = 1024 # No. of points in the y grid for evaluations,
yi = np.linspace(1E-4, 70, Ni)
# plot the analytic form of the PDF,
plt.plot(yi, y_pdf_analytic(yi, mu_sum, sqrt(var_sum)),'g', linewidth=2, label='Analytic')
plt.xlabel(r'$y$')
plt.ylabel('density')
plt.title(r'$x_1 = {0:.1f} \pm {1:.1f}, x_2 = {2:.1f} \pm {3:.1f}, \rho = {4:.2f}$'.format(*mus,*sigs,rho))
plt.ylim([0, 0.06])

# - the analytic moments are, 
y_mu_exact = y_moments_analytic(mu_sum, sqrt(var_sum), 1)
y_sigma_exact = sqrt(y_moments_analytic(mu_sum, sqrt(var_sum), 2) - y_mu_exact**2)

# 4. Now compare linear theory with the analytic result...
# - the linear mean of y is simply, 
y_mu_approx = mu_sum**2
y_sigma_approx = sqrt(4 * mu_sum**2 * var_sum)

# - for comparison on plot, 
yi_ext = np.linspace(-5, 70, Ni)
fapprox = gaussian(yi_ext, y_mu_approx, y_sigma_approx)
plt.plot(yi_ext, fapprox, 'r-', label='Linear Theory')
plt.legend()

print('* Sampling (mu, sigma) = {0:.2f}, {1:.2f}'.format(y_mu_samp, y_sigma_samp))
print('* Linear Theory (mu, sigma) = {0:.2f}, {1:.2f}'.format(y_mu_approx, y_sigma_approx))
print('* Exact (mu, sigma) = {0:.2f}, {1:.2f}'.format(y_mu_exact, y_sigma_exact))

# %% Compare with uncertainties package, 
import uncertainties as unc
x1, x2 = unc.ufloat(mu1, sigma1), unc.ufloat(mu2, sigma2)
y = (x1 + x2)**2 # 25. +/- 18.
print(' --- Uncertainties --- ')
print(f'* uncertainties (no correlation): y = {y}')
(x1, x2) = unc.correlated_values([mu1, mu2], covar)
y = (x1 + x2)**2 # 25. +/- 10.
print(f'* uncertainties (w/ correlation): y = {y}')


# %% Additional plot of the joint-pdf of our two random vars,
# - Plot of the bivariate, 
fig, ax = plt.subplots(1,1,num='Bivariate Normal', figsize=(5,5))
Z1, Z2 = np.meshgrid(np.linspace(mus[0] - 3*sigs[0], mus[0] + 3*sigs[0], 64),
                     np.linspace(mus[1] - 3*sigs[1], mus[1] + 3*sigs[1], 64))

points = np.vstack((np.ravel(Z1), np.ravel(Z2)))
PDF = multivariate_normal(points, mus, covar)
ax.contourf(Z1,Z2, PDF.reshape((64,64)), alpha=1)
ax.set_aspect('equal')
ax.set_ylabel(r'$x_2$', fontsize=16)
ax.set_xlabel(r'$x_1$', fontsize=16)
ax.set_title(r"Joint Distribution of $x$'s", fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
fig.tight_layout()
