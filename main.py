'''
title: "Introduction to the evreg package"
author: "Thierry Denoeux"
date: "`r Sys.Date()`"
output: rmarkdown::html_vignette
bibliography: tdenoeux.bib
vignette: >
  %\VignetteIndexEntry{Introduction to the evreg package}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}



The `evreg` package implements  ENNreg [@denoeux22b] [@denoeux23b], a neural network model for regression in which prediction uncertainty is quantified by Gaussian random fuzzy numbers (GRFNs), a newly introduced family of random fuzzy subsets of the real line that generalizes both Gaussian random variables and Gaussian possibility distributions [@denoeux23a]. The output GRFN is constructed by combining GRFNs induced by prototypes using a combination operator that generalizes Dempster's rule of Evidence Theory. The three output units indicate the most plausible  value of the response variable, variability around this value, and epistemic uncertainty. The network is trained by minimizing a loss function that generalizes the negative log-likelihood.

The `evreg` package contains functions for training the ENNreg model in batch or online mode, tuning hyperparameters by cross-validation or the hold-out method, and making predictions. It also contains utilities for making calculations with GRFNs (such as, e.g., computing the degrees of belief and plausibility of an interval, or combining GRFNs).

The user is invited to read the papers cited in this vignette to get familiar with the main concepts underlying epistemic random fuzzy sets and evidential regression. These papers can be downloaded from the author's web site, at <https://www.hds.utc.fr/~tdenoeux/>. Here, we provide a short guided tour of the main functions in the `evreg` package.
'''
###############
# You first need to install this package:

# ```{r, message=FALSE}
# library(evreg)
# ```
#########Install the python package


#############
# The following sections contain a brief introduction on the way to use the main functions in the package `evreg` for evidential regression.

## Evidential regression

### Data generation

# Let us start by writing a function that generates a dataset similar to that used in Section V.A of [@denoeux23b]:
import numpy as np
import random
from ENNreg_init import *
from ENNreg_cv import *
from ENNreg_ import *
from predict_ENNreg import *
from foncgrad_musigh import *
from foncgrad_RFS import *
from predict_ENNreg import *
from intervals import *
from Belint_ import *
from Bel import *
from Pl_ import *
from pl_contour import *
from rmsprop import *
from combination_GRFN import *
from ENNreg_holdout import *



def gendat(n):
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        u = random.uniform(0, 1)
        if u < 0.5:
            x[i] = random.uniform(-3, -1)
        else:
            x[i] = random.uniform(1, 4)

        if x[i] < 0:
            y[i] = np.sin(3 * x[i]) + x[i] + np.sqrt(0.01) * np.random.normal()
        else:
            y[i] = np.sin(3 * x[i]) + x[i] + np.sqrt(0.3) * np.random.normal()

    return {"x": x, "y": y}


# We generate training and test sets of sizes, respectively, 400 and 1000:
n = 400
nt = 1001

# Generate training data
train = gendat(n)

# Generate test data
test = gendat(nt)

# This Python code replicates the R code by setting the random seed using random.seed, generating training and test data using the gendat function, and using NumPy for numerical operations.


### Hyperparameter tuning and learning

# Let us determine hyperparameters $\xi$ and $\rho$ using cross-validation, with batch training and $K=30$ prototypes:
x = train['x']
y = train['y']

cv = ENNreg_cv(x, y, K=30, XI=[0, 0.01, 0.1], RHO=[0, 0.01, 0.1], verbose=False)
cv

# We can then train again the model using all the training data and the selected hyperparameters:


fit = ENNreg(train["x"], train["y"], K=30, xi=cv["xi"], rho=cv["rho"], verbose=False)

# Let us now compute the predictions for regularly spaced inputs:


xt = np.arange(-4, 5, 0.01)
pred = predict_ENNreg(fit, newdata=xt)

"""""
#and let us compute belief intervals at levels 50\%, 90\% and 99\%:

from scipy.stats import norm

# Example predicted values and their standard errors


# Given predict function (replace with your actual predict function)


# Confidence levels
confidence_levels = [0.50, 0.90, 0.99]

# Calculate prediction intervals
intervals = {}
for level in confidence_levels:
    # Calculate quantiles based on the standard normal distribution
    alpha = 1 - level
    z = np.abs(norm.ppf(alpha / 2))

    # Calculate prediction interval bounds
    lower_bound = pred - z * np.std(pred)
    upper_bound = pred + z * np.std(pred)

    intervals[f'int{int(level * 100)}'] = (lower_bound, upper_bound)

# Access the intervals as needed, e.g., intervals['int50'], intervals['int90'], intervals['int99']

"""""


# The intervals list now contains prediction intervals for the specified confidence levels
int50 = intervals(pred, 0.50)
int90 = intervals(pred, 0.9)
int99 = intervals(pred, 0.99)

import matplotlib.pyplot as plt

x = xt
mux = pred['mux']

# Plot the data points and prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(train['x'], y, label='Training Data', alpha=0.5)
plt.fill_between(x, int50["INTBel"][:, 0], int50["INTBel"][:, 1], alpha=0.2, label='50% Prediction Interval',
                 color='blue')#######interval
plt.fill_between(x, int90["INTBel"][:, 0], int90["INTBel"][:, 1], alpha=0.15, label='90% Prediction Interval',
                 color='green')#######interval
plt.fill_between(x, int99["INTBel"][:, 0], int99["INTBel"][:, 1], alpha=0.1, label='99% Prediction Interval',
                 color='orange')#######interval
plt.plot(x, mux, color='red', label='Predicted', linewidth=2)####red line
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Prediction Intervals')
plt.grid(True)
plt.show()

### Calibration curvesx

# Let us now plot calibration curves for probabilistic and belief prediction intervals. We start by computing the predictions for the test set:


test_x = test["x"]
test_y = test["y"]

# Make predictions on the test data
pred_tst = predict_ENNreg(fit, newdata=test_x.reshape(-1, 1))

# We then compute prediction intervals and their coverage rates for 9 equally spaced levels between 0.1 and 0.9:


# Define values for A
A = np.arange(0.1, 1.0, 0.1)
nA = len(A)

# Initialize arrays to store coverage probabilities
probbel = np.zeros(nA)
probp = np.zeros(nA)

# Calculate coverage probabilities for different values of A
for i in range(nA):
    # Assuming you have 'test_y' and 'pred_tst' already defined
    int_ = intervals(pred_tst,A[i],test['y'])

    # Calculate coverage probabilities for Bel and P
    probbel[i] = int_["coverage_Bel"]
    probp[i] = int_["coverage_P"]

# 'probbel' and 'probp' now contain the coverage probabilities for different levels of A


import numpy as np
import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(8, 6))

# Plot probp in blue
plt.plot([0] + list(A) + [1], [0] + list(probp) + [1], 'b-', lw=2, label='Probp', color='blue')

# Plot probbel in red
plt.plot([0] + list(A) + [1], [0] + list(probbel) + [1], 'r-', lw=2, label='Probbel', color='red')

# Add a dashed diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Diagonal', linestyle='--')

# Set labels and legend
plt.xlabel('Level')
plt.ylabel('Coverage Rate')
plt.title('Coverage Rates for Different Levels')
plt.legend(loc='lower right')

# Show the plot
plt.grid(True)
plt.show()

## Calculations with Gaussian random fuzzy numbers

# The `evreg` package also contains functions for computing with, and combining GRFNs. For instance, functions `Bel` and `Pl` compute, respectively, the degrees of belief and plausibility of intervals $[x,y]$. Let us illustrate the use of these functions for plotting $Bel([x-r,x+r])$ and $Pl([x-r,x+r])$ as functions of $x$, for different values of $r$. We start by defining the GRFN:


# Define parameters for GRFN
GRFN = {'mu': 1, 'sig': 1, 'h': 1}

x = np.arange(-4, 6, 0.01)
bel_1 = Bel(x - 1, x + 1, GRFN)
bel_05 = Bel(x - 0.5, x + 0.5, GRFN)
bel_01 = Bel(x - 0.1, x + 0.1, GRFN)

# We can then draw the "belief plot":
plt.plot(x, bel_1, label='r=1', linewidth=2)
plt.plot(x, bel_05, label='r=0.5', linewidth=2, linestyle='--')
plt.plot(x, bel_01, label='r=0.1', linewidth=2, linestyle=':')
plt.xlabel('x')
plt.ylabel('Bel([x-r,x+r])')
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()


# and the "plausibility plot"
pl_1 = Pl(x - 1, x + 1, GRFN)
pl_05 = Pl(x - 0.5, x + 0.5, GRFN)
pl_0 = pl_contour(x, GRFN)

plt.plot(x, pl_1, label='r=1', linewidth=2)
plt.plot(x, pl_05, label='r=0.5', linewidth=2, linestyle='--')
plt.plot(x, pl_0, label='r=0', linewidth=2, linestyle=':')
plt.xlabel('x')
plt.ylabel('Pl([x-r,x+r])')
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()


# We can also plot the lower cumulative distribution function
bel_values = Bel(-np.inf, x, GRFN)
pl_values = Pl(-np.inf, x, GRFN)

plt.plot(x, bel_values, label='Bel', linewidth=2)
plt.plot(x, pl_values, label='Pl', linewidth=2)
plt.xlabel('x')
plt.ylabel('Lower/upper cdfs')
plt.legend(loc='upper right')
plt.show()


# Comparison
GRFN1 = {'mu': 0, 'sig': 2, 'h': 4}
GRFN2 = {'mu': 1, 'sig': 1, 'h': 1}

GRFN12s = combination_GRFN(GRFN1, GRFN2, soft=True)
GRFN12h = combination_GRFN(GRFN1, GRFN2, soft=False)


pl_contour_GRFN1 = pl_contour(x, GRFN1)
pl_contour_GRFN2 = pl_contour(x, GRFN2)
pl_contour_GRFN12s = pl_contour(x, GRFN12s['GRFN'])
pl_contour_GRFN12h = pl_contour(x, GRFN12h['GRFN'])

plt.plot(x, pl_contour_GRFN1, label='GRFN1', linewidth=2, color='blue')
plt.plot(x, pl_contour_GRFN2, label='GRFN2', linewidth=2, linestyle='--', color='red')
plt.plot(x, pl_contour_GRFN12s, label='soft comb.', linewidth=2, linestyle=':', color='green')
plt.plot(x, pl_contour_GRFN12h, label='hard comb.', linewidth=2, linestyle=':', color='cyan')
plt.xlabel('x')
plt.ylabel('plausibility')
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()