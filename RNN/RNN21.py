# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:50:17 2018

@author: zhao
"""

import copy
import numpy as np

np.random.seed(0)

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_derivative(output):
    return output*(1-output)

#data generation
int2binary = {}
binary_dim = 8
largest_number = pow(2,binary_dim)
    # generate a 256*8 array of 0/1
binary = np.unpackbits(np.array([range(largest_number)],dtype = np.uint8).T,axis = 1)

for i in range(largest_number):
    int2binary[i] = binary[i]
    
# input

yita = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

#initiallize

u = 2*np.random.random((input_dim,hidden_dim)) - 1
v = 2*np.random.random((hidden_dim,output_dim)) - 1
w = 2*np.random.random((hidden_dim,hidden_dim)) - 1

u_gradient = np.zeros_like(u)
v_gradient = np.zeros_like(v)
w_gradient = np.zeros_like(w)

for j in range(100):
    #define relationship between x&y -> x+y
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int]
    b_int = np.random.randint(largest_number/2) 
    b = int2binary[b_int]
    # label
    c_int = a_int + b_int
    c = int2binary[c_int]
    # y_
    d = np.zeros_like(c)
    
    
    Err_All = 0
    
    layer_2_delta = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    for position in range(binary_dim):
        
        #input&output generation
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
        #feedforward
        # h(t) = ux+wh(t-1)
        h = sigmoid(np.dot(X,u)+np.dot(layer_1_values[-1],w))
        #output = vh(t)
        y_ = sigmoid(np.dot(h,v))
        # compute err
        layer_2_err = y-y_
    
    
    
    
    
    
    
    
    
    
    