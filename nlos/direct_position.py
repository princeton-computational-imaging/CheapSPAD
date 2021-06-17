# Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
# Author: Zheng Shi

# direct position prediction approach
import argparse
import os
import numpy as np
import math
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, Sequential
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt


def net(hist_len, size = 50, depth = 5):
    # distance network
    g2 = tf.Graph()
    with g2.as_default() as g:
        x = tf.placeholder(tf.float32, [None, 4, hist_len, 1])
        y = tf.placeholder(tf.float32, [None, 3])
        is_train = tf.placeholder(tf.bool)

        fc = tf.layers.flatten(x)
        for i in range(depth): 
            fc = tf.layers.dense(fc, size, activation = tf.nn.relu)
        fc = tf.layers.dense(fc, 3)
        
        y_pred = tf.layers.flatten(fc)
        
        loss = tf.reduce_mean(tf.square(y - y_pred))
        diff = tf.reduce_mean(tf.reduce_sum((y - y_pred)**2, axis = 1))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

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
    parser.add_argument('--mat_file', default = './data/bigtarget_mirrors.mat.mat', type = str, help = 'path to the measurement')
    parser.add_argument('--ckpt_path', default = './checkpoint/mirror_big_distance', type=str, help='dir to save trained model')
    parser.add_argument('--with_mirror', default = True, type=str2bool, help='whether mirror is included in the setup')
    
    args = parser.parse_args()

    # set paramters---------------------------------------------------------------
    X_key = 'datacubes'
    P_key = 'positions'
    num_measurements = 2 # numbers of histograms to average 
    seed = 111 # seed used for train_test_split
    test_size = 0.2 # seed used for train_test_split
    remove_firstn = 6 # trim the first n bins to filter out reflection from the mirror

    # load and process data---------------------------------------------------------------
    mat= scipy.io.loadmat(args.mat_file)
    X = mat[X_key][0][:] # measurement
    P = mat[P_key][0,:] # target location
    X_train, X_test, P_train, P_test = train_test_split(X, P, test_size=test_size, random_state=seed) # fix seed for reproducibility

    x_train = []
    y_train = []
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
            datai /= np.sum(np.mean(X_train[i][:,:,:remove_firstn,(j)*num_measurements:(j+1)*num_measurements], axis=-1))
            datai = np.multiply(datai, np.arange(1,24-remove_firstn)**2)
            x_train.append(np.reshape(datai, (4,-1)))
            y_train.append(P_train[i])

                                    
    x_train = np.asarray(x_train)
    x_train = x_train[...,np.newaxis]
    y_train = np.asarray(y_train)
    y_train = y_train.squeeze()

    x_test = []
    y_test = []

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
            datai /= np.sum(np.mean(X_test[i][:,:,:remove_firstn,(j)*num_measurements:(j+1)*num_measurements], axis=-1))
            datai = np.multiply(datai,np.arange(1,24-remove_firstn)**2)
            x_test.append(np.reshape(datai, (4,-1)))
            y_test.append(P_test[i])
        
    x_test = np.asarray(x_test)
    x_test = x_test[...,np.newaxis]
    y_test = np.asarray(y_test)
    y_test = y_test.squeeze()

    # network ---------------------------------------------------------------
    # define network 
    g_net, x_net, y_net, y_pred_net, is_train_net, loss_net, diff_net, optimizer_net = net(x_train.shape[2], 16, 5)
    sess_net = tf.Session(graph=g_net)
    if args.cmd == 'train':
        # initialize network
        with g_net.as_default():
            sess_net.run(tf.global_variables_initializer())
        list_train_loss_net = []
        list_test_loss_net = []
        # training
        for i in range(3000):
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
    print("Test RMSE: %.03f" %(np.sqrt(np.mean((y_pred_net_ - y_test) **2))))
    print("Prediction std: ", np.std(y_pred_net_,axis=0))

if __name__ == '__main__':
    main()