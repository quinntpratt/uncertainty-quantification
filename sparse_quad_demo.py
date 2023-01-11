#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:13:56 2021

@author: Quinn Pratt

Investigating sparse quadrature rules for numerical integration and 
uncertainty quantification.

We use the Smolyak construction based on 1D quadratures and dimension-recursion.

To avoid large amounts of cancellation we use the combinatorical methods 
outlined in the thesis by V. Kaarnioja (2013)
see: https://core.ac.uk/download/pdf/14928756.pdf

You can also see this Sandia Labs presentation about Smolyak quadratures,
https://www.osti.gov/servlets/purl/1514349

To do: 
    - include "delayed" sequences
    - further benchmaking (ChaosPy, Sandia UQTk)
    - normalize weights to sum to 1?
    - track down factor of 2 in model problem (see line 433)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from math import sqrt, pi, comb
import quadpy
from scipy.spatial import KDTree

class SmolyakQuad(object):
    def __init__(self, d, order, method='gauss_hermite'):
        ''' 
        Initialization of a d-dimensional smolyak sparse quadrature of 
        order k.
        
        Parameters
        ----------
        d : int
            Dimensionality of quadrature.
        order : int
            Order parameter (related to the accuracy of the quadrature)
        method : str
            Method to generate the 1D quadratures,
                [1] gauss_hermite
    
        Stores as attrs
        ---------------
        nodes : np.ndarray
            Matrix of quadrature points with shape: (N-by-d) where N is the final
            number of nodes in the sparse quadrature.
        weights : np.ndarray
            Vector of weights corresponding to each point in the quadrature.
            Note: weigths need not be positive.
        '''
        self.dim, self.order = d, order
        self.method = method
        
        snodes = []
        sweights = []
        # - bounds for outer summation,
        k = order + d
        l_min, l_max = max([d, k-d+1]), k
        # - iterate over the l param (due to behavior of range we must add one
        #   to actually iterate with l = k)
        for l in range(l_min, l_max + 1):
            # - generate the list of multi-indicies for this level "l", 
            inds = list(self.get_multi_inds(d,l))
            #print(f'* l = {l} alphas: {inds}')
            # - for each multi-index (mi) tuple in the above list,
            for mi in inds:
                # - the mi has length = d and indicates univariate quadrature
                #   'order' to be used along each dimension
                #   e.g. mi = (1,2,1) says to use an order 1 quad along dim1
                #   an order 2 quad along dim 2, and an order 1 quad along dim3
                
                # - we store each univariate quadrature in a list of length d
                nodes = [None] * d
                weights = [None] * d
                for ji, j in enumerate(mi):
                    nodes[ji], weights[ji] = self.get_1D_quad(j, method=self.method)
                
                # - mesh-grid the list of nodes and reshape into a [N-by-d]
                #   array where each ROW is a quadrature point.
                x = np.stack(np.meshgrid(*nodes), -1).reshape(-1, d)
                # - perform the same reshape/grid with the weights, and then 
                #   product along the dimension-axis so we have only 1 weight 
                #   for each node.
                _w = np.stack(np.meshgrid(*weights), -1).reshape(-1, d)
                w = self._coeff(k, d, l) * np.prod(_w, axis=1)
                
                snodes += [x]
                sweights += [w]
        
        
        # store the nodes and weights as attributes,
        self.nodes = np.concatenate(snodes, axis=0)
        self.weights = np.concatenate(sweights, axis=0)
        self.cost = len(self.weights)
        
        # refactor nodes for nested schemes,
        self.refactor_nested()
        
        # normalize weights to 1.
        self.weights /= np.sum(self.weights)
        
    def _coeff(self, k, d, l):
        return (-1)**(k-l)*comb(d-1, k-l)
    
    def get_multi_inds(self, d, l):
        '''
        Function to generate all d-dimensional multi-indicies "alpha" under the 
        constraints, 
            (1) alpha >= the d-dimensional one-vector
            (2) The one-norm (sum) of alpha is equal to l
        
        This function is based on a recursive method,
            https://stackoverflow.com/a/29171375/14663456
        with small modification to satisfy (1) above.
    
        Parameters
        ----------
        d : int
            Dimensionality of system.
        l : int
            Order (one-norm) parameter.
    
        Returns
        -------
        This function returns a generator for these tuples.
        e.g.
        >> list(get_multi_inds(2,3))
        >> [(1, 2), (2, 1)]
        '''
        
        if d == 1:
            yield (l,)
            return
    
        for i in range(1,l):
            for t in self.get_multi_inds(d - 1, l - i):
                yield (i,) + t
                
    def get_1D_quad(self, k, method='gauss_hermite'):
        ''' 
        Method to generate the nodes and weights for various 1D quadrature rules
        We use the syntax from the module "quadpy" found at, 
            https://pypi.org/project/quadpy/
        
        We default to UQ-related quadratures (weighted by e^-x^2) so we assume
        we are fetching a method from the e1r2 group of quadratures.
        
        Note: quadpy.e1r2.gauss_hermite is equivalent to,
            np.polynomial.hermite.hermgauss(k)
        
        This function must return np.ndarray-type vectors: points, weights.
        '''
        try:
            scheme = getattr(quadpy.e1r2, method)(k)
        except AttributeError:
            scheme = getattr(quadpy.c1, method)(k)
            
        return scheme.points, scheme.weights
    
    def refactor_nested(self, thresh=1E-5):
        '''
        Function to re-factor the quadrature and group points/weights which 
        are generated from nested methods.
        
        The threshold parameter sets the radius around points for merging.
        
        It appears that this extra step is needed due to minute numerical 
        differences between points and should not have a measureable impact on
        the result.
        '''
        nodes = self.nodes
        weights = self.weights
        # - generate a KDTree object from the nodes, 
        nodes_tree = KDTree(nodes)
        # - find pairs within thresh (works with 3-wise pairs)
        inds_to_nest = nodes_tree.query_pairs(thresh,output_type='ndarray')
        # - further sorting,
        inds_sort = np.argsort(inds_to_nest[:,-1])
        inds_to_nest = inds_to_nest[inds_sort,:]
        # - loop backwards through this array to merge all...
        deleted_inds = []
        for i in range(len(inds_sort)-1,-1,-1):
            # - get the inds to sort (the second one appears always larger)
            j, k = inds_to_nest[i,:]
            if k in deleted_inds:
                continue
            # - the weights are summed,
            weights[j] += weights[k]
            # - delete the corresponding 'extra' row from the nodes array,
            nodes = np.delete(nodes, k, axis=0)
            # - delete the 'extra' weight from the weights array,
            weights = np.delete(weights, k)
            # - store the deleted index,
            deleted_inds += [k]
            
        # - update the quadrature, 
        self.nodes = nodes
        self.weights = weights
        self.cost = len(self.weights)
        self.refactored = True
        
        return self.nodes, self.weights
        
    
    def affine_map(self, A, b):
        ''' 
        Function to apply an affine transformation to scale quadratures
        from the "standard" space --> the *real* domain for integration.
            y = Ax + b
        where,
            x : quadrature nodes (N-by-d)
            A : scale/rotation matrix (d-by-d)
            b : shift, length-d vector 
        
        e.g. performing a 1D integral from 2 --> 5 would require,
            A = [1.5], b = [3.5] to map a quadrature from [-1, 1] --> [2, 5]
        
        e.g. perforing a 2D integral over the square [2,5] x [2,5] given
            a standard-space quadrature [-1, 1] x [-1, 1] requires, 
            A = 1.5*np.eye(2), and b = np.array([3.5, 3.5])
            
        This is a utility included for UQ purposes because we often need to 
        scale by the mean and sqrt(cov-matrix)
        
        see: https://en.wikipedia.org/wiki/Affine_transformation
        '''
        x = self.nodes.T # shape: (d x N)
        return (A @ x).T + b
        
        
    def quad_plot(self, ax=None, scale=200):
        ''' 
        Aux. method to plot quadratures in 1, 2, and 3 dimensions
        
        In 2D red = negative-weights, blue = positive-weights.
        '''
        if self.dim > 3:
            print('! cannot plot quadratures in dim > 3')
            return
        if ax is None:
            if self.dim < 3:
                fig, ax = plt.subplots(1,1,num='SmolyakQuad: quad_plot')
            else:
                fig = plt.figure(num='SmolyakQuad: quad_plot')
                ax = fig.add_subplot(projection='3d')
        
        if self.dim == 1:
            ax.stem(self.nodes, self.weights)
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Weight')
        else:
            cmap = ListedColormap(['red', 'blue'])
            cbool = self.weights > 0
            norm = BoundaryNorm([0, 0.5, 1], cmap.N)
            
        if self.dim == 2:
            ax.scatter(self.nodes[:,0], self.nodes[:,1],
                       s=scale*abs(self.weights/max(self.weights)),
                       c=cbool,
                       cmap=cmap,
                       edgecolor='k',
                       alpha=0.5,
                       norm=norm)
            ax.set_aspect('equal')
        elif self.dim == 3:
            ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:,2],
                       s=scale*abs(self.weights/max(self.weights)),
                       c=cbool,
                       cmap=cmap,
                       edgecolor='k',
                       alpha=0.5,
                       norm=norm)
        return ax
                       
# dimension,
d = 2
# order,
order = 2
# generate the quadrature:
method = "gauss_hermite"
quad = SmolyakQuad(d, order, method=method)
nodes, weights = quad.nodes, quad.weights
# Use the plotting method,
ax = quad.quad_plot(scale=350)
ax.grid()

# %% Benchmarking,
# Compare with the chaospy module, 
# see: https://chaospy.readthedocs.io/en/master/api/chaospy.generate_quadrature.html
# and: https://www.sintef.no/globalassets/project/evitameeting/2015/feinberg_lecture2.pdf
import chaospy as cp

# To use the chaospy generate_quadrature method we need to supply, 
#   (a) the order
#   (b) the distribution
# This is a little clunky when it comes to multivariate problems (dim > 1).
# we need to create a "Joint" distribution using the chaospy.J method.

if method == "gauss_hermite":
    args = [cp.Normal(0, 1/sqrt(2)) for i in range(d)]
    dist = cp.J(*args)
    rule = "G"
elif method == "gauss_legendre":
    args = [cp.Uniform(-1,1) for i in range(d)]
    dist = cp.J(*args)
    rule = 'E'

cp_nodes, cp_weights = cp.generate_quadrature(order, dist, rule=rule, sparse=True)
# plot these nodes, 
if d == 2:
    ax.plot(cp_nodes[0,:], cp_nodes[1,:], 'k *', ms=6)
elif d == 3:
    ax.scatter(cp_nodes[0,:], cp_nodes[1,:], cp_nodes[2,:], color="k", marker="*",)

# CONCLUSION: we have the same weights and nodes as the chaospy library.

# %% - Application to model problem,
# In this section we compare the quadrature method for a simple 2D UQ example.

# Consider two uncertain inputs, (X1, X2) with a joint normal distribution,
#   (X1, X2) ~ N(mu, covar)

# What is the distrbution of Y1, Y2 given,
#   Y1 = np.sqrt(X1^2 + X2^2)
#   Y2 = np.atan2(X2/X1)

# i.e. the nonlinear process is the conversion to polar coordinates.

# To approach this with sparse quadratures we will, 
#   0. Declare the properties of the Joint X1, X2 distribution.
#   1. Generate a sparse quadrature 
#   2. Evaluate the nonlinear function over our quadrature.
#   3. Compute the w.avg using the results from 2. and the weights from 1.
#   4. Compute the w.var using "" "" ""  

# %% 0. Setup

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

def non_lin_func(x):
    ''' non-linear function for testing UQ
    x: array-type with shape: (N, d) where N: No. of pts, d: No. of dims
    out: same dims as x
    '''
    out = 0.*x
    out[:, 0] = np.linalg.norm(x, axis=1)
    out[:, 1] = np.arctan2(x[:, 1], x[:, 0])
    return out


# - Vector of means,
mu = np.array([1, 2])
# - Vector of Std.Devs,
sig = np.array([0.2, 0.15])
# - Correlation coeff, (for 2D only one is given) 
rho = -0.75
# - Covariance Matrix, 
Sigma = np.array( [[sig[0]**2, rho*sig[0]*sig[1]],
                   [rho*sig[0]*sig[1], sig[1]**2]])
print(" --- Sparse Quad. Demonstration --- ")
print(f'* Vector of Means: {mu} ')
print(f'* Vector of Std.Devs.: {sig} ')
print("* Covariance Matrix, ")
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

# %% 1. Generate appropriate sparse quadrature

quad = SmolyakQuad(2, 2, method="gauss_hermite")
weights = quad.weights
N = len(weights)
# NOTE: these quadrature points lie in the standard-normal subspace.
# they must be brought to the problem-space via an affine transformation.

# - basically the matrix square-root of the covar matrix,
A = np.linalg.cholesky(Sigma)
# - shifted by the mean,
b = mu
# - creates a new set of points.
nodes = quad.affine_map(A, b)

ax.plot(nodes[:,0],nodes[:,1], 'r o')

# %% 2-4 Use the quadrature, 

# 2. Evaluate,
y = non_lin_func(nodes)

# 3. Compute the 1st moment (mean output)
# (a) using the numpy average function,
y_avg = np.average(y, weights=weights,axis=0)
# (b) by explicit sum,
y_avg = np.sum(y.T*weights, axis=1)/np.sum(weights)

# 4. Compute the 2nd moment (covariance of output)
# WARNING: Why do we multiply by 2 here? It's wrong unless we do?
e = y - y_avg 
y_covar = 2*sum( [weights[i] * np.outer(e[i,:], e[i,:]) for i in range(N)] )

y1_std = sqrt(y_covar[0,0])
y2_std = sqrt(y_covar[1,1])

print(" --- Results --- ")
print(f"y1 = {y_avg[0]:.4f} +/- {y1_std:.4f}")
print(f"y2 = {y_avg[1]:.4f} +/- {y2_std:.4f}")


# %% Compare with Monte Carlo,
# No. of points for MC,
N_samples = 10000
x_samp = np.random.multivariate_normal(mu, Sigma, size=N_samples)
# - pipe all of these samples through our non-linear function,
y_samp = non_lin_func(x_samp)

# - compute the mean, and std. of each, 
y_avg_mc = np.average(y_samp, axis=0)
y_std_mc = np.std(y_samp, axis=0)

print(f"vs. MonteCarlo with N = {N_samples}: ")
for i in [0, 1]:
    print('y{0} = {1:.4f} +/- {2:.4f}'.format(i+1, y_avg_mc[i], y_std_mc[i]))