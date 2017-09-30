#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:24:19 2017

@author: yellow
"""
import numpy as np
import kernel_regression as kr
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin

class KernelRegression_OSCV(kr.KernelRegression):
    def _optimize_gamma(self, gamma_values):
        C = 0.5730 # rescalling parameter for Gaussian Kernel OSCV
        A_0 = np.empty([len(self.X),len(self.X)]) # set A matrix to save comparison results of x_i x_j
        B_0 = np.empty([len(self.X),len(self.X)])
        A_1 = np.empty([len(self.X),len(self.X)])
        B_1 = np.empty([len(self.X),len(self.X)])
        C_0 = np.empty([len(self.X),len(self.X)])
        C_1 = np.empty([len(self.X),len(self.X)])
        for i in xrange(len(self.X)):
            for j in xrange(len(self.X)):
                A_0[i][j] = self.X[i][0]-self.X[j][0]
                B_0[i][j] = (A_0[i][j] >= 0).astype(int)
                C_0[i][j] = (A_0[i][j] < 0).astype(int)
                A_1[i][j] = self.X[i][1]-self.X[j][1]    
                B_1[i][j] = (A_1[i][j] >= 0).astype(int)
                C_1[i][j] = (A_0[i][j] < 0).astype(int)
       
        mse_0 = np.empty_like(gamma_values, dtype=np.float)
        mse_1 = np.empty_like(gamma_values, dtype=np.float)
        
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, self.X, metric=self.kernel,gamma=gamma)
    
            K_0 = np.multiply(2*K, B_0)
            K_0 = np.multiply(K_0, B_1)
            K_1 = np.multiply(2*K, C_0)
            K_1 = np.multiply(K_1, C_1)
            np.fill_diagonal(K_0, 0)  # leave-one-out
            np.fill_diagonal(K_1, 0)  # leave-one-out
            Ky_0 = K_0 * self.y[:, np.newaxis]
            Ky_1 = K_1 * self.y[:, np.newaxis]
          
            y_pred_0 = Ky_0.sum(axis=0) / K_0.sum(axis=0)
            y_pred_1 = Ky_1.sum(axis=0) / K_1.sum(axis=0)

            mse_0[i] = np.nanmean((y_pred_0 - self.y) ** 2)
            mse_1[i] = np.nanmean((y_pred_1 - self.y) ** 2)
            
        
        gamma_hat_0 = C*gamma_values[np.nanargmin(mse_0)]
        gamma_hat_1 = C*gamma_values[np.nanargmin(mse_1)]
            
        return (gamma_hat_0 + gamma_hat_1)/2.0
        