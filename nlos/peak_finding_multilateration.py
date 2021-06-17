# Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
# Author: Zheng Shi

# peak finding and multilateration approach
import argparse
import os
import numpy as np
import math
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, Sequential
import scipy.io
from scipy.spatial import distance
from scipy.spatial import distance
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

def shift_hist(data, start_idx, hist_len):
    # shift histogram so that the direct peak happens at the first bin, and pad the resulted historgram with 0 to hist_len
    xp = np.arange(data.shape[-1])
    fp = data
    hist_ = np.interp(np.arange(0,data.shape[-1],0.1), xp, fp)[start_idx:]
    hist_ = np.pad(hist_, (0, hist_len*10-hist_.shape[-1]), 'constant', constant_values=(0,0))    
    return np.mean(hist_.reshape(-1, 10), 1)

def x2dist(x, hist_len, b1 = 0, b2 = 6):
    # b1: looking for direct peak starting from b1
    # b2: looking for indirect peak starting from b2 after shifting the histogram
    dist = 0.2*np.ones((4,1)) # one bin equaiv to 40cm round trip distance / 20 cm one way
    # estimate direct peak location using weighted sum
    direct_peak_idx = np.sum(x[:,b1:,0] * np.tile(np.arange(x[:,b1:,0].shape[-1]), (4,1)), -1) / np.sum(x[:,b1:,0], -1)
    # estimate indirect peak location using weighted sum, after intensity conpensation
    for i in range(4):
        hist_ = np.multiply(shift_hist(x[i,b1:,0], int(direct_peak_idx[i]*10), hist_len),np.arange(1,hist_len + 1)**2)
        indirect_peak_idx = np.sum(hist_[b2:] * np.arange(hist_[b2:].shape[-1]), -1) / np.sum(hist_[b2:])
        dist[i] *= indirect_peak_idx
    return dist

def peak_finding_multilateration(x, hist_len, b1, b2, train_loc_mean):
    # given measurement x, peak finding and multilaterating to the target location
    dist = x2dist(x,hist_len, b1, b2)
    tl = [1.06-1.67, 0, 1.53]
    bl = [1.06-1.60, 0, 1.08]
    tr = [1.06-0.52, 0, 1.99]
    br = [1.06-0.57, 0, 1.21]
    def loss(Y):
        return np.abs(distance.euclidean(Y, tl) - dist[0]) +\
    np.abs(distance.euclidean(Y, tr) - dist[1]) +\
    np.abs(distance.euclidean(Y, bl) - dist[2]) +\
    np.abs(distance.euclidean(Y, br) - dist[3])  
    
    # minimize loss, initial point and boundary computed from training data
    y = minimize(loss, x0 = train_loc_mean, bounds = ((-0.5, 0.6), (-1.7, -0.5),(0.4,1.8))).x
    return y

def main():
    parser = argparse.ArgumentParser(
        description='Peak finding and multilateration approach for NLOS tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser.add_argument('--mat_file', default = './data/bigtarget_mirrors.mat', type = str, help = 'path to the measurement')
    parser.add_argument('--with_mirror', default = True, type=str2bool, help='whether mirror is included in the setup')
    
    args = parser.parse_args()

    # set paramters---------------------------------------------------------------
    X_key = 'datacubes'
    Y_key = 'dists_wall_obj'
    P_key = 'positions'
    num_measurements = 2 # numbers of histograms to average 
    seed = 111 # seed used for train_test_split
    test_size = 0.2 # seed used for train_test_split
    hist_len = 24 # pad the result histogram to hist_len after shifting it to start with direct peak

    if args.with_mirror:
        b1 = 6 # looking for direct peak after b1
        b2 = 6 # looking for indirect peak after b2 (after shift histogram)
    else:
        b1 = 0
        b2 = 3

    # load and process data---------------------------------------------------------------
    mat= scipy.io.loadmat(args.mat_file)
    X = mat[X_key][0][:] # measurement
    Y = mat[Y_key][0,:] # distance between wall and target object
    P = mat[P_key][0,:] # target location
    X_train, X_test, Y_train, Y_test, P_train, P_test = train_test_split(X, Y, P, test_size=test_size, random_state=seed) # fix seed for reproducibility

    x_test = []
    p_test = []

    for i in range(len(X_test)):
        if P_test[i][1] < -1.7:
            continue
        if X_test[i].ndim > 3 and X_test[i].shape[3] > 2:
            testhisto1 = X_test[i][0,1,9:,5]
            testhisto2 = X_test[i][1,1,9:,5]
            nhists = math.floor((X_test[i].shape[3] - 1) / num_measurements)
        else: 
            testhisto1 = X_test[i][0,1,9:]
            testhisto2 = X_test[i][1,1,9:]
            nhists = 1
            X_test[i] = X_test[i].reshape(2,2,24,-1)
        if args.with_mirror:
            if np.sum(testhisto1) < 20000 or np.sum(testhisto2) < 20000:
                continue

        for j in range(nhists):
            datai = np.mean(X_test[i][:,:,:,(j)*num_measurements:(j+1)*num_measurements], axis=-1)
            x_test.append(np.reshape(datai, (4,-1)))
            p_test.append(P_test[i])
        
    x_test = np.asarray(x_test)
    x_test = x_test[...,np.newaxis]
    p_test = np.asarray(p_test)
    p_test = p_test.squeeze()



    # check performance on test data ---------------------------------------------------------------
    y_loc = np.zeros_like(p_test)
    train_loc_mean = np.mean(P_train).squeeze()
    for i in range(p_test.shape[0]):
        y_loc[i,:] = peak_finding_multilateration(x_test[i], hist_len, b1, b2, train_loc_mean)
    print("Test RMSE: %.03f" %(np.sqrt(np.mean((y_loc - p_test) **2))))
    print("Prediction std: ", np.std(y_loc,axis=0))

if __name__ == '__main__':
    main()