# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 2021
@author: Kopal Garg
"""
import numpy as np
import pandas as pd
import os
import sys
import json
import numpy as np
import pickle as pkl

def load_simulated_sinusoid(data_path):

    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
            x_train = pkl.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
            x_test = pkl.load(f)

    return x_train, x_test