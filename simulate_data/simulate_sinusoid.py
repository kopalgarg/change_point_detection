import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
import random
from sklearn.model_selection import train_test_split

def main(type, num_samples, seq_length):
    signal_in = []

    for i in range(num_samples):
        if type=='irregularly_sampled_sinusoid':
            x = irregularly_sampled_sinusoid(seq_length=seq_length)
            signal_in.append(x)

    print(type, 'saved')
    signal_in = np.array(signal_in)

    # split 70% train 30% test
    x_train, x_test = train_test_split(signal_in, test_size=0.30, random_state=42)
    
    # print shape
    print("x_train shape", x_train.shape)
    print("x_test shape", x_test.shape)

    # save a sample plot
    plt.plot(x_train[0])
    
    return x_train, x_test, plt

def irregularly_sampled_sinusoid(seq_length):

    white_noise = ts.noise.GaussianNoise(std=0.3)
    sinusoid = ts.signals.Sinusoidal(frequency=0.25)
    time_series = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    sample, signals, error = time_series.sample(np.array(range(seq_length)))

    return np.array(sample)


def save_data(path,array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

if __name__=='__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    n_samples = 10 # samples per unit
    s_len = 50 # seq length

    type = 'irregularly_sampled_sinusoid'
    x_train, x_test, plt = main(type = type, num_samples= n_samples, seq_length = s_len)
    
    path = os.path.join('./data/', type)
    if not os.path.exists(path):
        os.mkdir(path)
    
    save_data(os.path.join(path, 'x_train.pkl'), x_train)
    save_data(os.path.join(path, 'x_test.pkl'), x_test)
    plt.savefig(os.path.join(path, 'sample_signal.png'))