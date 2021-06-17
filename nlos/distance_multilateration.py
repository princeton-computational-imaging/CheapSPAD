# Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
# Author: Zheng Shi

# 2 stage distance prediction and multilateration approach
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

def multilateration(dist, train_loc_mean):
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

def net(hist_len, size = 16, depth = 5):
    # distance network
    g2 = tf.Graph()
    with g2.as_default() as g:
        x = tf.placeholder(tf.float32, [None, hist_len, 1])
        y = tf.placeholder(tf.float32, [None, 1])
        is_train = tf.placeholder(tf.bool)

        fc = tf.layers.flatten(x)
        for i in range(depth): 
            fc = tf.layers.dense(fc, size, activation = tf.nn.relu)
        fc = tf.layers.dense(fc, 1)
        
        y_pred = tf.layers.flatten(fc)
        
        loss = tf.reduce_mean(tf.square(y - y_pred))
        diff = loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        return g2, x, y, y_pred, is_train, loss, diff, optimizer

def main():
    parser = argparse.ArgumentParser(
        description='Distance_Multilateration approach for NLOS tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser.add_argument('cmd', default='train', type=str,  choices=['train', 'test'],  help='train or test')
    parser.add_argument('--mat_file', default = './data/bigtarget_mirrors.mat', type = str, help = 'path to the measurement')
    parser.add_argument('--ckpt_path', default = './checkpoint/mirror_big_distance', type=str, help='dir to save trained model')
    parser.add_argument('--with_mirror', default = True, type=str2bool, help='whether mirror is included in the setup')
    
    args = parser.parse_args()

    # set paramters---------------------------------------------------------------
    X_key = 'datacubes'
    Y_key = 'dists_wall_obj'
    P_key = 'positions'
    num_measurements = 2 # numbers of histograms to average 
    seed = 111 # seed used for train_test_split
    test_size = 0.2 # seed used for train_test_split
    remove_firstn = 6 # trim the first n bins to filter out reflection from the mirror

    # load and process data---------------------------------------------------------------
    mat= scipy.io.loadmat(args.mat_file)
    X = mat[X_key][0][:] # measurement
    Y = mat[Y_key][0,:] # distance between wall and target object
    P = mat[P_key][0,:] # target location
    X_train, X_test, Y_train, Y_test, P_train, P_test = train_test_split(X, Y, P, test_size=test_size, random_state=seed) # fix seed for reproducibility

    x_train = []
    y_train = []
    p_train = []
    for i in range(len(X_train)):
        if P_train[i][1] < -1.7:
            continue
        if X_train[i].ndim > 3 and X_train[i].shape[3] > 2:
            testhisto1 = X_train[i][0,1,9:,5]
            testhisto2 = X_train[i][1,1,9:,5]
            nhists = math.floor((X_train[i].shape[3] - 1) / num_measurements)
        else: 
            testhisto1 = X_train[i][0,1,9:]
            testhisto2 = X_train[i][1,1,9:]
            nhists = 1
            X_train[i] = X_train[i].reshape(2,2,24,-1)
        if args.with_mirror:
            if np.sum(testhisto1) < 20000 or np.sum(testhisto2) < 20000:
                continue

        for j in range(nhists):
            datai = np.mean(X_train[i][:,:,remove_firstn:-1,(j)*num_measurements:(j+1)*num_measurements], axis=-1)
            direct_peak_idx = np.sum(datai * np.tile(np.arange(datai.shape[-1]), (2,2,1)), -1) / np.sum(datai, -1)
            datai /= np.sum(np.mean(X_train[i][:,:,:remove_firstn,(j)*num_measurements:(j+1)*num_measurements], axis=-1))
            
            for x in range(2):
                for y in range(2):
                    hist = datai[x,y,:]
                    hist = np.multiply(hist,np.arange(1,hist.shape[-1] + 1)**2)
                    x_train.append(hist)
                    y_train.append(Y_train[i][x,y])
            p_train.append(P_train[i])

                                
    x_train = np.asarray(x_train)
    x_train = x_train[...,np.newaxis]
    y_train = np.asarray(y_train)
    y_train = y_train[...,np.newaxis]
    p_train = np.asarray(p_train).squeeze()

    x_test = []
    y_test = []
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
            datai = np.mean(X_test[i][:,:,remove_firstn:-1,(j)*num_measurements:(j+1)*num_measurements], axis=-1)
            direct_peak_idx = np.sum(datai * np.tile(np.arange(datai.shape[-1]), (2,2,1)), -1) / np.sum(datai, -1)
            datai /= np.sum(np.mean(X_test[i][:,:,:remove_firstn,(j)*num_measurements:(j+1)*num_measurements], axis=-1))
            
            for x in range(2):
                for y in range(2):
                    hist = datai[x,y,:]
                    hist = np.multiply(hist,np.arange(1,hist.shape[-1] + 1)**2)
                    x_test.append(hist)
                    y_test.append(Y_test[i][x,y])
            p_test.append(P_test[i])

        
    x_test = np.asarray(x_test)
    x_test = x_test[...,np.newaxis]
    y_test = np.asarray(y_test)
    y_test = y_test[...,np.newaxis]
    p_test = np.asarray(p_test).squeeze()

    # network ---------------------------------------------------------------
    # define network 
    g_net, x_net, y_net, y_pred_net, is_train_net, loss_net, diff_net, optimizer_net = net(x_train.shape[1], 16, 5)
    sess_net = tf.Session(graph=g_net)
    if args.cmd == 'train':
        # initialize network
        with g_net.as_default():
            sess_net.run(tf.global_variables_initializer())
        list_train_loss_net = []
        list_test_loss_net = []
        # training
        for i in range(1000):
            idx = np.random.randint(0,len(x_train), 32)
            X_in = x_train[idx]
            Y_in = y_train[idx]
            _,l_train_net_, _ = sess_net.run([loss_net, diff_net, optimizer_net], feed_dict={x_net: X_in, y_net:  Y_in, is_train_net: True})
            y_pred_net_,l_test_net_ = sess_net.run([y_pred_net, diff_net], feed_dict={x_net: x_test, y_net:  y_test, is_train_net: False})
            list_train_loss_net.append(l_train_net_)
            list_test_loss_net.append(l_test_net_)  
        # save checkpoint
        with g_net.as_default():
            saver = tf.train.Saver()
            saver.save(sess_net, args.ckpt_path)
    else:
        # load pre-trained network
        with g_net.as_default():
            sess_net.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess_net, args.ckpt_path)

    # check performance on test data ---------------------------------------------------------------
    y_pred_net_,l_test_net_ = sess_net.run([y_pred_net, diff_net], feed_dict={x_net: x_test, y_net:  y_test, is_train_net: False})
    y_loc = np.zeros_like(p_test)
    train_loc_mean = np.mean(P_train).squeeze()
    for i in range(int(y_pred_net_.size/4)):
        y_loc[i,:] = multilateration(y_pred_net_[4*i:4*i + 4].squeeze(), train_loc_mean)
    print("Test RMSE: %.03f" %(np.sqrt(np.mean((y_loc - p_test) **2))))
    print("Prediction std: ", np.std(y_loc,axis=0))

if __name__ == '__main__':
    main()