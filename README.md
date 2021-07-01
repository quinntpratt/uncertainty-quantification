# Tutorials in Uncertainty Quantification
Python programs demonstrating various aspects of uncertainty quantification/propagation.

## Description

This project consists of several tutorial scripts demonstrating uncertainty quantification (UQ) and uncertainty propagation calculations using simple models. The goal here is *not* to create a flexible UQ toolbox - rather to illustrate some low-level mathematical results for pedagogy.

The three scripts currently pertain to **linear error propagation theory**, but in the future we intend to include quadrature-based methods and PCE.

## Getting Started

### Dependencies

All programs were developed in ``python 3.X``. 

I have assumed the user has standard scientific python packages such as ``numpy``, ``matplotlib`` (standard to an Anaconda distribution for example) with the addition of the ``uncertainties`` package for comparisons with linear error-propagation theory, and the ``lmfit`` package (for the fit example). These packages are available here, 

* [uncertainties](https://pythonhosted.org/uncertainties/user_guide.html)
* [lmfit](https://lmfit.github.io/lmfit-py/)


## Help

If you run into a problem running the scripts - it's most likely a dependency issue, see above.

## Authors

Quinn Pratt
