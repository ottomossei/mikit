import os, io, sys, re
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern, ExpSineSquared
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import numpy as np

from .compname import ChemFormula, TriChemFormula

BASIC_KERNEL = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()

class BayesOpt():
    def __init__(self, X_all, xi = 0.01, kernel = None):
        """
        Args:
            X_all(numpy) : Search area
            xi(float) : Trade-off parameter between exploration and exploitation
            kernel(sklearn) : Kernel used in gaussian distribution
        """
        self.X_all = X_all
        self.xi = xi
        self.kernel = kernel if kernel is not None else BASIC_KERNEL

    def fit(self, X_exp, y_exp):
        """
        Args:
            X_exp(numpy) : Explanatory variables
            y_exp(numpy) : Objective variable
        """
        y_exp = y_exp.reshape(-1, 1)
        scaler_y = StandardScaler().fit(y_exp)
        # Fitting of gaussian process
        gpr = GaussianProcessRegressor(kernel = self.kernel)
        y_trans = scaler_y.transform(y_exp)
        gpr.fit(X_exp, y_trans)
        # Prediction of fitting model
        mu, sigma = gpr.predict(self.X_all, return_std = True)
        mu = scaler_y.inverse_transform(mu)
        mu_exp, mu_exp_std = gpr.predict(X_exp, return_std = True)
        mu_exp = scaler_y.inverse_transform(mu_exp)
        # Calculate the ei
        sigma = sigma.reshape(-1, 1)
        mu_exp_opt = np.max(mu_exp)
        with np.errstate(divide = 'warn'):
            imp = mu - mu_exp_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        self.mu = mu.ravel()
        self.sigma = sigma.ravel()
        self.ei = ei.ravel()
    
    def get_info(self):
        return self.mu, self.sigma, self.ei
    
    def delete_cutoff(self, idx, X_obj, ei):
        """
        Args:
            idx(int) : Idex of next point
            X_obj(numpy) : Explanatory variables
            ei(numpy) : Expectation improvement
        Return:
            X_obj(numpy) : Explanatory variables without variables in the cutoff radius
            ei(numpy) : Expectation improvement without variables in the cutoff radius
        """
        p = X_obj[idx]
        idx_del = list()
        for i in range(ei.shape[0]):
            q = X_obj[i]
            if np.sqrt(np.power(p-q, 2).sum()) < self.cutoff:
                idx_del.append(i)
        X_obj = np.delete(X_obj, idx_del, 0)
        ei = np.delete(ei, idx_del, 0)
        return X_obj, ei
        
    def get_next(self, num, cutoff=0.1):
        """
        Args:
            num(int) : The number of the next point
            cutoff(float) : Cutoff distance in Euclidean space
        Return:
            X_next : Recommended coordinates (next point)
        """
        self.cutoff = cutoff
        ei = self.ei.copy()
        X_obj = self.X_all.copy()
        X_next = list()
        for n in range(num):
            idx = ei.argmax()
            X_next.append(X_obj[idx])
            X_obj, ei = self.delete_cutoff(idx, X_obj, ei)
        return np.array(X_next)
