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
        mean, variance, wasst_dist, univariate_2sample, multivariate_2sample = compute_metrics(series=x_train[i,:,:], prev_window_len=prev_window_len, next_window_len=next_window_len)


def compute_metrics(series, prev_window_len, next_window_len):
    mean = []
    variance = []
    wasst_dist = []
    univariate_2sample = []
    multivariate_2sample = []

    for i in range(prev_window_len, series.shape[1]-next_window_len):
        prev = series[:, i-prev_window_len:i]
        next = series[:, i:i+next_window_len]
        
    return mean, variance, wasst_dist, univariate_2sample, multivariate_2sample

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection based on correlation structure and distribution changes')
    parser.add_argument('--type', type=str, default='irregularly_sampled_sinusoid')
    parser.add_argument('--out_path', type=str, default='./out')
    parser.add_argument('--prev_window_len', type = int, default = 2)
    parser.add_argument('--next_window_len', type = int, default = 2)
    args = parser.parse_args()

    main(out_path = args.out_path, type = args.type, prev_window_len = args.prev_window_len, next_window_len = args.next_window_len)

        

