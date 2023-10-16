# coding: utf-8
import sys, os
sys.path.append(os.pardir)  
import numpy as np
from functions import *
from functions import _delsigmoid


class TwoLayerNet_sigmoid:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01,lr=0.1,num_pt=21):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.predictions = []
        self.lr = lr
        self.num_pt = num_pt

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        
        return y
        
    def loss(self, x, t):
        y = self.predict(x)
        error = sum_squared_error(y, t)
        return error
    
    def accuracy(self, x, t):
        y = self.predict(x)
        class_y = np.zeros((self.num_pt, 1))
        for idx in range(0,self.num_pt):
          if y[idx]>=0.5:
            class_y[idx] = 1
        self.predictions = class_y
        accuracy = np.sum(class_y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        # forward
        a1 = np.dot(x, W1) + b1 #hidden
        z1 = sigmoid(a1)  #hidden_out
        a2 = np.dot(z1, W2) + b2  #output_
        y = sigmoid(a2) #output_final
        # y is the output_final 
        
        # backward
        error_term = (t - y)
        grads['W1'] = x.T @ (((error_term * _delsigmoid(y)) * W2.T) * _delsigmoid(z1))
        grads['W2'] = z1.T @ (error_term * _delsigmoid(y))
        grads['b1'] = np.sum(self.lr * ((error_term * _delsigmoid(y)) * W2.T) * _delsigmoid(z1), axis=0)
        grads['b2'] = np.sum(self.lr * error_term * _delsigmoid(y), axis=0)

        return grads