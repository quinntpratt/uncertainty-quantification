#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:49:55 2021

@author: Quinn Pratt

This script is to explore the Scaled Unscented Transform (SUT) as a means 
of propagating uncertain inputs through a nonlinear process (function).

The (S)UT is often part of a filtering procedure, such as a Kalman filter.
We will use it here for uncertainty quantification (UQ).

This was NOT presented as part of Plasmapy Hack-Week 2021.

Additional dependancy: filterpy 
for benchmarking our simplex-points against an existing module.
"""

import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
#from filterpy.kalman import SimplexSigmaPoints

def symmetric_sigma(mu, covar, beta=None, zero_pt=True):
    ''' Function to return sigma points and weights according to the 
    symmetric generation process.
    '''
    # - dimension of the problem
    n = len(mu)
    # - if no scaling is provided, use the original "unscented option"
    if beta is None:
        beta = sqrt(n)
        zero_pt = False
    # - compute the cholesky decomp, 
    sqSig = np.linalg.cholesky(covar)
    # - perpare the 2n symmetric points,
    # -- the points are stored in an n-by-2n matrix... each column is a sigma pt.
    sig_x = np.zeros((n,2*n))
    sig_w = 1/(2*beta**2) * np.ones(2*n)
    # - generate the points, 
    for i in range(n):
        sig_x[:,i] = mu + beta * sqSig[:,i]
        sig_x[:,i+n] = mu - beta*sqSig[:,i]
    
    # - if we are to compute the "zero-point" concatenate it onto the front, 
    if zero_pt:
        wo = 1. - n/(beta**2)
        xo = mu
        sig_x = np.hstack([ np.atleast_2d(xo).T, sig_x ])
        sig_w = np.concatenate([ [wo], sig_w])
    
    # - return the generated sigma points, 
    return sig_x, sig_w

def simplex_sigma(mu, covar, w0=0.0):
    ''' Function to return sigma points and weights according to the 
    spherical-simplex algorithm
    
    Based on the algorithm in Fig. 2 of, 
    [1] S. Julier "The Spherical Simplex Unscented Transformation" (2003)
    
    But, it was pointed out in, 
    [2] H. Menegaz, "A new smallest sigma set for the UT" (2011)
    
    that the mean and covarainance matrix produced here are not equal to the
    mean and covariance of the *prior* distribution (for dim >= 1).
    
    Note, FilterPy uses a simplex algorithm from, 
    [3] P. Moireau, "Reduced-Order UKF" (2010)
    http://www.numdam.org/article/COCV_2011__17_2_380_0.pdf
    
    which claims to have sourced their algorithm from a 2002 Julier paper...
    but [2] claims that this algorithm also has problems. However, upon closer
    inspection, it appears that the algorithm of [3] is not the same as the 2002
    julier paper - therefore it may not necessarily have the same issues.
    
    '''
    # - dimension of the problem
    n = len(mu)
    # - number of sigma points (including the "zeroth" point) 
    Npts = n + 2 
    # - given the initial weight (w0), we construct the other weights...
    
    sig_w = (1 - w0)/(1 + n) * np.ones(Npts-1)
    sig_w = np.concatenate([ [w0], sig_w ])
    # - preallocate the sigma points,
    #   this is an [N_dim x N_pts] matrix of sigma point pos. vectors,
    sig_x = np.zeros((n, Npts))
    
    # - Init. the 0th, 1st, and 2nd sigma-pts for the *1st* dim. 
    # (the zero-th one is just zero, so it's already initialized w/ prelloc.)
    sig_x[0,1] = -1/sqrt(2*sig_w[1])
    sig_x[0,2] = 1/sqrt(2*sig_w[1])
    # - after this initialization, we have populated just the first element of 
    #   each sigma-point-position vector
    
    # - for the remaining dimensions (elments in the sigma-pt pos-vects)
    #   we "expand" the sequence of vectors as per, 
    for j in range(2,n+1):
        # for the 0th sigma point they're just zero... so leave this alone
        # because we already preallocated with zeros. 
        
        # for sigma points running from the 2nd dim. (index = 1) up through
        # the jth dimension (index = j) (remember range is *not* end-point inclusive) 
        for i in range(1, j+1):
            sig_x[j-1,i] = -1/sqrt(j*(j+1)*sig_w[1])
        
        # for the final, (j+1)th sigma point (index i = j + 1)
        sig_x[j-1,j+1] = j/sqrt(j*(j+1)*sig_w[1])


    # - Plot-testing, 
    # plt.figure()
    # plt.plot(sig_x[0,:], sig_x[1,:], 'b o', ms=10)
    # circ = Circle((0,0),radius=sqrt(n),facecolor='none', edgecolor='r')
    # plt.gca().set_aspect('equal')
    # plt.gca().add_artist(circ)
    
    # - Use the mean and covariance matrix to map this simplex into our space,
    # - compute the cholesky decomp, (this is an N_dim-by-N_dim matrix) 
    sqSig = np.linalg.cholesky(covar)
    # - perform an affine transformation, 
    #   we matmul a [Nd x Nd] w/ a [Nd x Npts] 
    #   the result is an [Nd x Npts] matrix (same as sig_x)
    #   then, to each sigma point, we add the means, 
    sig_x = np.atleast_2d(mu).T + np.dot(sqSig, sig_x)
    
    # - return the generated sigma points, 
    return sig_x, sig_w

def min_sigma(mu, covar, w0=0.5):
    ''' "Minimal" Sigma Point set proposed by
    [2] H. Menegaz, "A new smallest sigma set for the UT" (2011)
    '''
    # - dimensionality,
    n = len(mu)
    
    # 1. Use w0 to construct the weights matrix,
    # - alpha is a scalar,
    alpha = sqrt((1 - w0)/n)
    # - C is an [n x n] matrix constructed as follows:
    C2 = np.eye(n) - alpha**2*np.ones((n, n))
    #   C is the matrix-square-root of this, 
    C = np.linalg.cholesky(C2)
    
    # - To compute the weights we need other related quantities, 
    #   The inverse of C,
    invC = np.linalg.inv(C)
    #   The inverse of the transpose of C,
    invCT = np.linalg.inv(C.T)
    
    # - The "full" weights matrix is constructed via several matrix products, 
    Wfull = w0*alpha**2 * invC.dot( np.ones((n,n)).dot(invCT) )
    # - The "W" matrix is the diagonal,
    #   note: calling diag() created a vector... call diag() again to build the 
    #         proper matrix, 
    W = np.diag(Wfull)
    # - the weights are: the 0th weight tacked onto the diagonal of W,
    sig_w = np.concatenate( [[w0], W] )
    
    # 2. The points are constructed as per:
    # - we need the sqrt of the covar matrix, 
    sqrt_covar = np.linalg.cholesky(covar)
    # - the 0th point is special,
    #   dot -sqrt_covar with an [n x 1] column vector of alpha/sqrt(w0)
    x0 = -sqrt_covar.dot( alpha*np.ones((n,1))/sqrt(w0))
    
    # - To compute the other n points we need this matrix,  
    inv_sqrt_W = np.linalg.inv( np.diag(np.sqrt(W)) )
    #   then, dot sqrt_covar with C and it, 
    xn = sqrt_covar.dot( C.dot(inv_sqrt_W) )
    
    # - concatenate and add the mean vector, 
    sig_x = np.atleast_2d(mu).T + np.c_[x0, xn]
    
    # - The result is an [n x n+1] sigma-point matrix (points are columns)
    
    return sig_x, sig_w

def bivariate_normal(x, mu, covar):
    ''' Function to compute the n-dimensional multivariate normal given
    a vector of mu's and the covar-matrix.
    :arg x: n-by-M set of M points in nD over which to evaluate
    :arg mu: length n vector of centers
    :arg covar: [n x n] covariance matrix
    '''
    n, M = np.shape(x)
    z = np.zeros(M)
    for i in range(M):
        z[i] = (x[:,i] - mu).T @ (np.linalg.inv(covar) @ (x[:,i] - mu))
    prefactor = (2*pi*np.linalg.det(covar))**(-1/2)
    return prefactor*np.exp(-0.5*z)

# 0. Setup
# - Vector of means,
mu = np.array([1, 2])
# - Vector of Std.Devs,
sig = np.array([0.2, 0.15])
# - Correlation coeff, (for 2D only one is given) 
rho = -0.75
# - Correlation Matrix, 
Sigma = np.array( [[sig[0]**2, rho*sig[0]*sig[1]],
                   [rho*sig[0]*sig[1], sig[1]**2]])
print(" --- (S)UT Demonstration --- ")
print(f'* Vector of Means: {mu} ')
print(f'* Vector of Std.Devs.: {sig} ')
print("* Correlation Matrix, ")
print(Sigma)

# - Plot the 2D PDF 
fig, ax = plt.subplots(1, 1)
fs = 16
Z1, Z2 = np.meshgrid(np.linspace(mu[0] - 3*sig[0], mu[0] + 3*sig[0], 64),
                     np.linspace(mu[1] - 3*sig[1], mu[1] + 3*sig[1], 64))

points = np.vstack((np.ravel(Z1), np.ravel(Z2)))
PDF = bivariate_normal(points, mu, Sigma)
ax.contourf(Z1,Z2, PDF.reshape((64,64)), alpha=1)
half_max = 0.5*(2*pi*np.linalg.det(Sigma))**(-1/2)
ax.contour(Z1,Z2, PDF.reshape((64,64)),[0.2*half_max, half_max],colors='b', linestyles='--')
ax.set_aspect('equal')
ax.set_ylabel(r'$x_2$', fontsize=fs)
ax.set_xlabel(r'$x_1$', fontsize=fs)
ax.set_title('Joint Distribution of Xs')

# %% 1. Generation of sigma point-sets,
# Given the properties of the input distribution (mu and Sigma), we are 
# now able to construct various sets of sigma points and their corresponding 
# weights. There are various methods for constructing this pointset,
#   (a) the "symmetric" method.
#   (b) the "simplex" method.
#   (c) *several others*

# (a) - The symmetric points,
sig_x, sig_w = symmetric_sigma(mu, Sigma)
Npts = len(sig_w)
print(" --- "*3)
print('* Symmetric Sigma points, ')
print(sig_x)
print('* Symmetric Sigma weights, ')
print(sig_w)
print(" --- "*3)
# plot them 
ax.scatter(sig_x[0,:], sig_x[1,:], 500*sig_w,
           color='r', alpha=0.6, edgecolor='k', label="symmetric")

# (b) - The simplex points,
ssig_x, ssig_w = simplex_sigma(mu, Sigma, w0=0.1)
Npts_simplex = len(ssig_w)
print('* Simplex Sigma points, ')
print(ssig_x)
print('* Simplex Sigma weights, ')
print(ssig_w)
print(" --- "*3)
# plot,
ax.scatter(ssig_x[0,:], ssig_x[1,:], 500*ssig_w,
           color='b', alpha=0.6, edgecolor='k', label="simplex")

# Testing, 
# sig_pts_obj = SimplexSigmaPoints(2)
# ssig_w = sig_pts_obj.Wm
# ssig_x = sig_pts_obj.sigma_points(mu,Sigma).T

# (c) - The "minimal" points,
msig_x, msig_w = min_sigma(mu,Sigma,w0=0.5)
Npts_min = len(msig_w)
print('* Minimal Sigma points, ')
print(msig_x)
print('* Minimal Sigma weights, ')
print(msig_w)
print(" --- "*3)
ax.scatter(msig_x[0,:], msig_x[1,:], 500*msig_w,
           color='orange', alpha=0.6, edgecolor='k', label="minimal")

# plot dressing, 
ax.legend()

# %% 2. Use the (S)UT to perform uncertainty propogation,
#   to make use of the S(UT) we need a nonlinear function to propagate our 
#   two uncertain inputs (x1, x2) through.
#
#   We will compare the distribution of outputs with MonteCarlo unc. prop..
# 
#   For demo purposes we will use a 2-input, 2-output nonlinear function, 
#       Y1, Y2 ~ F(X1, X2)
#   we will use the conversion from cartesian (X1, X2) to polar (Y1, Y2)
#   defined by the function "non_lin_func(X)" below.
#   i.e. Y1 = sqrt( X1^2 + X2^2 ); Y2 = atan2(X2/X1)

#   In the limit that X1 and X2 are uncorrelated/equal-variance 
#       Y1 ~ Rician distribution (cf. https://en.wikipedia.org/wiki/Rice_distribution)
#       Y2 ~ *unnamed* distribution

def non_lin_func(x):
    ''' non-linear function for testing UQ
    x: array-type with shape: (N, d) where N: No. of pts, d: No. of dims
    out: same dims as x
    '''
    out = 0.*x
    out[:, 0] = np.linalg.norm(x, axis=1)
    out[:, 1] = np.arctan2(x[:, 0], x[:, 1])
    return out

# 2.1 pass the sigma-points through the function,
y = non_lin_func(sig_x.T) # shape: (Npts, dims)
# for comparisons we will also pass the simplex and minimal point-sets through,
y_simplex = non_lin_func(ssig_x.T)
y_minimal = non_lin_func(msig_x.T)

# 2.2 compute the average over the outputs,
# NOTE: for all point-sets this is just the weighted-average,
y_avg = np.average(y, weights=sig_w, axis=0) # shape: (dims)
y_avg_simplex = np.average(y_simplex, weights=ssig_w, axis=0)
y_avg_min = np.average(y_minimal,weights=msig_w, axis=0)

# 2.3 compute the output covariance,
#   this is the sought-after quantity when it comes to UQ.
#   since we have two outputs, (Y1, Y2), we expect a 2x2 covar matrix.
# NOTE: there are different formulas for the different point-sets - be careful!

# - deviation w.r.t. average for each point,
e = y - y_avg # shape: (N, dims)
es = y_simplex - y_avg_simplex
em = y_minimal - y_avg_min

# - Take an outer-product and perform a weighted average over 2x2 matricies,
y_covar = sum( [sig_w[i]*np.outer(e[i,:], e[i,:]) for i in range(Npts)] )
y_covar_simplex = sum( [ssig_w[i]*np.outer(es[i,:], es[i,:]) for i in range(Npts_simplex)] )
y_covar_min = sum( [msig_w[i]*np.outer(em[i,:], em[i,:]) for i in range(Npts_min)] )

# 2.4 Print the results,
print(" --- Results --- ")
print('* symmetric, ')
for i in [0, 1]:
    print('y{0} = {1:.3f} +/- {2:.3f}'.format(i+1, y_avg[i], sqrt(y_covar[i,i])))

print('* simplex, ')
for i in [0, 1]:
    print('y{0} = {1:.3f} +/- {2:.3f}'.format(i+1, y_avg_simplex[i], sqrt(y_covar_simplex[i,i])))
    
print('* minimal, ')
for i in [0, 1]:
    print('y{0} = {1:.3f} +/- {2:.3f}'.format(i+1, y_avg_min[i], sqrt(y_covar_min[i,i])))

# %% 3. Compare the (S)UT results with other methods,
#   In this section we compare the UQ results from the (S)UT with 
#   'standard' Monte Carlo error propagation.

# No. of points for MC,
N_samples = 10000
x_samp = np.random.multivariate_normal(mu, Sigma, size=N_samples)
# - pipe all of these samples through our non-linear function,
y_samp = non_lin_func(x_samp)
# Produce a histogram,
# fig2, (ax1, ax2) = plt.subplots(1,2)
# ax1.hist(y_samp[:, 0], bins=20, edgecolor='k', alpha=0.5, color='b')
# ax1.set_xlabel('y1', fontsize=fs)
# ax2.hist(y_samp[:, 1], bins=20, edgecolor='k', alpha=0.5, color='g')
# ax2.set_xlabel('y2', fontsize=fs)

# - compute the mean, and std. of each, 
y_avg_mc = np.average(y_samp, axis=0)
y_std_mc = np.std(y_samp, axis=0)

print(f"vs. MonteCarlo with N = {N_samples}: ")
for i in [0, 1]:
    print('y{0} = {1:.3f} +/- {2:.3f}'.format(i+1, y_avg_mc[i], y_std_mc[i]))

# Analytic Comparison,
# NOTE - this assumes X1 and X2 are uncorrelated (i.e. set rho = 0)
#       and they have the same variance (i.e. set sig = np.array([s, s]))

from scipy.special import hyp1f1

# parameters for the Rice distribution,
nu = np.linalg.norm(mu)
s = sig[0]
K = nu**2/2/s**2
# analytic results for the mean and std.dev of Y1,
y1_analytic_avg = s*sqrt(pi/2)*hyp1f1(-0.5,1,-K)
y1_analytic_std = np.sqrt( 2*s**2 + nu**2 - pi*s**2/2*( hyp1f1(-0.5,1,-K) )**2 )

print(f"vs. Analytic (given rho = 0, sigma = [s,s ...], ")
print(f"y1 = {y1_analytic_avg:.3f} +/- {y1_analytic_std:.3f}")

# The angle distribution has an analytic form but no simple expression for the
# first two moments.