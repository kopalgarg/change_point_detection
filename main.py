# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 2021
@author: Kopal Garg
"""
from load_data import load_simulated_sinusoid
import numpy as np
import pandas as pd
import os
import sys
import json
import numpy as np
import argparse
from matplotlib import pyplot as plt
from distribution_change import *
from correlation_change import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main(out_path, type, prev_window_len, next_window_len):

    if args.out_path=='./out' and not os.path.exists('./out'):
        os.mkdir('./out')
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # load the data
    if type in ['irregularly_sampled_sinusoid']:
        data_path = os.path.join('./data/', type)
        x_train, x_test = load_simulated_sinusoid(data_path)

    # split series into before and after windows
    n_samples = x_train.shape[0]

    for i in range(n_samples):
        mean_vals = mean(series=x_train[i,:,:], prev_window_len=prev_window_len, next_window_len=next_window_len, aggregate=True, threshold=2)
        variance_vals = variance(series=x_train[i,:,:], prev_window_len=prev_window_len, next_window_len=next_window_len, aggregate=True, threshold=1)


def mean(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    mean_diffs = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :]
        mean_diff = abs(np.average(prev, axis = 0) - np.average(next, axis = 0))
        if aggregate==True:
            if mean_diff >= threshold:
                mean_diffs.append(mean_diff)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                mean_diffs.append(mean_diff)
        else:
            temp_len = prev_window_len
            mean_diffs.append(mean_diff)
    return mean_diffs

def variance(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    var_diffs = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :]
        var_diff = abs(np.variance(prev, axis = 0) - np.variance(next, axis = 0))
        if aggregate==True:
            if var_diff >= threshold:
                var_diffs.append(var_diffs)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                var_diffs.append(var_diff)
        else:
            temp_len = prev_window_len
            var_diffs.append(var_diff)
    return var_diffs

def univariate_KS(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    t_vals = []
    p_vals = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :]
        if prev.any() and next.any():
            p_val, t_val = univariate_ks(prev, next)
        if aggregate==True:
            if t_val >= threshold:
                
                t_vals.append(t_val)
                p_vals.appned(p_val)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                t_vals.append(t_val)
                p_vals.append(p_val)
        else:
            temp_len = prev_window_len
            t_vals.append(t_val)
            p_vals.append(p_val)
    return t_vals, p_vals

def multivariate_MMD_linear(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    mmd_linear_vals = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :] 
        mmd_linear_val = mmd_linear(prev, next)
        if aggregate==True:
            if mmd_linear_val >= threshold:
                mmd_linear_vals.append(mmd_linear_val)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                mmd_linear_vals.append(mmd_linear_val)
        else:
            temp_len = prev_window_len
            mmd_linear_vals.append(mmd_linear_val)
    return mmd_linear_vals

def multivariate_MMD_poly(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    mmd_poly_vals = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :] 
        mmd_poly_val = mmd_poly(prev, next)
        if aggregate==True:
            if mmd_poly_val >= threshold:
                mmd_poly_vals.append(mmd_poly_val)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                mmd_poly_vals.append(mmd_poly_val)
        else:
            temp_len = prev_window_len
            mmd_poly_vals.append(mmd_poly_val)
    return mmd_poly_vals

def multivariate_MMD_rbf(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    mmd_rbf_vals = []
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :] 
        mmd_rbf_val = mmd_rbf(prev, next)
        if aggregate==True:
            if mmd_rbf_val >= threshold:
                mmd_rbf_vals.append(mmd_rbf_val)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                mmd_rbf_vals.append(mmd_rbf_val)
        else:
            temp_len = prev_window_len
            mmd_rbf_vals.append(mmd_rbf_val)
    return mmd_rbf_vals

def multivariate_wasserstein(series, prev_window_len=10, next_window_len=10, aggregate=True, threshold=2):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    ws=[]
    temp_len = prev_window_len
    for i in range(prev_window_len,series.shape[0]-next_window_len,1):
        i = int(i)
        prev = series[i-temp_len:i, :]
        next = series[i:i+next_window_len, :] 
        dist, _, _ = sinkhorn(prev, next)
        if aggregate==True:
            if dist >= threshold:
                ws.append(dist)
                temp_len = prev_window_len
            else:
                temp_len = temp_len+prev_window_len
                ws.append(dist)
        else:
            temp_len = prev_window_len
            ws.append(dist)
    return ws
    
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--type', type=str, default='irregularly_sampled_sinusoid')
    parser.add_argument('--out_path', type=str, default='./out')
    parser.add_argument('--prev_window_len', type = int, default = 2)
    parser.add_argument('--next_window_len', type = int, default = 2)
    args = parser.parse_args()

    main(out_path = args.out_path, type = args.type, prev_window_len = args.prev_window_len, next_window_len = args.next_window_len)

        

