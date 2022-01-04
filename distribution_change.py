# -*- coding: utf-8 -*-
"""
Created on Mon Jan 3 2022
@author: Kopal Garg
"""
import numpy as np
import pandas as pd
import os
import sys
import json
import numpy as np
import pickle as pkl
from scipy.stats import ks_2samp
import metrics
import torch
import torch.nn as nn

def univariate_ks(prev, next):
    """
    Univariate 2 Sample Testing with Bonferroni Aggregation

    Arguments:
        prev {vector} -- [n_sample1, dim]
        next {vector} -- [n_sample2, dim]
    Returns:
        p_val -- [p-value]
        t_val -- [t-value, i.e. KS test-statistic]

    """
    p_vals = []
    t_vals = []

  # for each dimension, we conduct a separate KS test
    for i in range(prev.shape[1]):
        feature_tr = prev[:, i]
        feature_te = next[:, i]

        t_val, p_val = None, None
        t_val, p_val = ks_2samp(feature_tr, feature_te)
        p_vals.append(p_val)
        t_vals.append(t_val)

    # apply the Bonferroni correction for the family-wise error rate by picking the minimum
    # p-value from all individual tests
    p_vals = np.array(p_vals)
    t_vals = np.array(t_vals)
    p_val = min(np.min(p_vals), 1.0)
    t_val = np.mean(t_vals)
  
    return p_val, t_val


def mmd_linear(X, Y):
    """
    MMD with Linear Kernel

    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
    Returns:
        mmd_val -- [MMD value]
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    mmd_val=XX.mean() + YY.mean() - 2 * XY.mean()

    return mmd_val

def mmd_rbf(X, Y, gamma=1.0):
    """
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    
    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
        gamma {float} -- [kernel parameter, default: 1.0]
    
    Returns:
        mmd_val {scalar} -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """
    MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    
    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
        degree {int} -- [degree, default: 2)
        gamma {int} -- [gamma, default: 1]
        coef0 {int} -- [constant item, default: 0]

    Returns:
        mmd_val {scalar} -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

class SinkhornDistance(nn.Module):
    r"""
    
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Arguments:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.shape[-1] == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + np.expand_dims(u,-1) + np.expand_dims(v,-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = torch.from_numpy(np.expand_dims(x,-2))
        y_lin = torch.from_numpy(np.expand_dims(y,-3))
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

