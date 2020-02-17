# Automated Hyperparameter Tuning using Bayesian Optimization 

In this notebook, we use Bayesian Optimization to automatically tune hyperparameters for 2 popular Machine Learnng packages:
* Xgboost, the top gradient boosting package, and
* SVR (Support Vector Regressor) part of SVM, the Support Vector Machine family

**Data:**               We will use the 'Diabetes' dataset that is included in the sklearn package

**Scoring:**             MSE (Mean square Error) is the metric we have chosen

**Validation strategy:** We will use Cross-validation, to estimate the accuracy of our models

**Performance Gain:**    Upon Tuning, we will compute the performance gain as follows: Perforance Gain = (Baseline score(no tuning) / Achieved score) * 100. So if Baseline MSE is 100 and we achieve an MSE of 50, we would have obtained a 100% performance gain over the baseline (no tuning, default parameter values)



! pip install GPy gpyopt xgboost

## Import Packages 

import numpy as np

import GPy

import GPyOpt

import matplotlib

import matplotlib.pyplot as plt

import sklearn

from sklearn.svm import SVR

import sklearn.datasets

import xgboost

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

import time

%matplotlib inline

## Package versions

for p in [np, GPy, GPyOpt, sklearn, xgboost, matplotlib]:
    print (p.__name__, p.__version__)

## Load Data

dataset = sklearn.datasets.load_diabetes()
X = dataset['data']
y = dataset['target']

## 1- Xgboost

Parameters tuned:
*  max_depth, 
* learning_rate,
* n_estimators,
* min_child_weight, and
* gamma

#### Score. Optimizer will try to find minimum, so we will add a "-" sign.
def f(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
        XGBRegressor(learning_rate=parameters[0],
                     max_depth=int(parameters[2]),
                     n_estimators=int(parameters[3]),
                     gamma=int(parameters[1]),
                     min_child_weight = parameters[4]), 
        X, y, scoring='neg_mean_squared_error'
    ).mean()
    score = np.array(score)
    return score

baseline = -cross_val_score(
    XGBRegressor(), X, y, scoring='neg_mean_squared_error'
).mean()
baseline

#### Bounds (NOTE: define continuous variables first, then discrete!)
bounds = [
    {'name': 'learning_rate',
     'type': 'continuous',
     'domain': (0, 1)},

    {'name': 'gamma',
     'type': 'continuous',
     'domain': (0, 5)},

    {'name': 'max_depth',
     'type': 'discrete',
     'domain': (1, 50)},

    {'name': 'n_estimators',
     'type': 'discrete',
     'domain': (1, 300)},

    {'name': 'min_child_weight',
     'type': 'discrete',
     'domain': (1, 10)}
]

np.random.seed(777)
optimizer = GPyOpt.methods.BayesianOptimization(
    f=f, domain=bounds,
    acquisition_type ='MPI',
    acquisition_par = 0.1,
    exact_eval=True
)

max_iter = 50
max_time = 60
optimizer.run_optimization(max_iter, max_time)


optimizer.plot_convergence()

Best values of parameters:

optimizer.X[np.argmin(optimizer.Y)]

print('MSE:', np.min(optimizer.Y),
      'Gain:', baseline/np.min(optimizer.Y)*100)

**Result:**   We obtained a 7.7% performance gain over the baseline (default parameter values, and no tuning)

## 2- SVR (Support Vector Regressor)

Parameters tuned:
* C, 
* epsilon, and
* gamma

#### Score. Optimizer will try to find minimum, so we will add a "-" sign.
def f(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
        SVR(C=parameters[0],
            epsilon=float(parameters[1]),
            gamma=float(parameters[2])), 
        X, y, scoring='neg_mean_squared_error'
    ).mean()
    score = np.array(score)
    return score

baseline = -cross_val_score(
    SVR(), X, y, scoring='neg_mean_squared_error'
).mean()
baseline

#### Bounds (NOTE: define continuous variables first, then discrete!)
bounds = [
    {'name': 'C',
     'type': 'continuous',
     'domain': (1e-5, 1000)},

    {'name': 'epsilon',
     'type': 'continuous',
     'domain': (1e-5, 10)},

    {'name': 'gamma',
     'type': 'continuous',
     'domain': (1e-5, 10)}
]

np.random.seed(777)
optimizer = GPyOpt.methods.BayesianOptimization(
    f=f, domain=bounds,
    acquisition_type ='MPI',
    acquisition_par = 0.1,
    exact_eval=True
)

max_iter = 50
max_time = 60
optimizer.run_optimization(max_iter, max_time)

optimizer.plot_convergence()

Best value of parameters

#### Best value of parameters
optimizer.X[np.argmin(optimizer.Y)]

print('MSE:', np.min(optimizer.Y),
      'Gain:', baseline/np.min(optimizer.Y)*100)

**Result:**  We achieved a 70.9% gain over the baseline (default parameter values and no tuning)
