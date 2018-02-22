# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:53:01 2018

@author: zhao
"""
from __future__ import division
import numpy as np
np.random.seed(1)
yita = 0.3
# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
x_test = np.array(([4,8]), dtype=float)
# scale units
X = X/np.amax(X, axis=0) # maximum of X array
x_test = x_test/np.amax(x_test, axis=0) # maximum of xPredicted (our input data for the prediction)
y = y/100 # max test score is 100
# three layder BP

class NN(object):
    def __init__(self):
        #parameters
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3
        #weights
        self.w1 = np.random.randn(self.input_size,self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size,self.output_size)
     
    def forward(self,X):
        
        self.z1 = np.dot(X,self.w1)
        self.a1 = self.sigmoid(self.z1) #activation
        self.z2 = np.dot(self.a1,self.w2)
        output = self.sigmoid(self.z2)# final activation
        
        return output
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
        
    def sigmoid_prime(self,s):
        return s*(1-s)
        
    def backword(self,X,y,output):
        self.output_err = (output-y)*(1/3)
        self.output_delta = self.output_err*self.sigmoid_prime(output)
        
        self.a1_err = self.output_delta.dot(self.w2.T)
        self.a1_delta = self.a1_err*self.sigmoid_prime(self.a1)
        
        self.w1 -= yita*X.T.dot(self.a1_delta)
        
        self.w2 -= yita*self.a1.T.dot(self.output_delta)
        
    def train(self,X,y):
        output = self.forward(X)
        self.backword(X,y,output)
    def predict(self):
        
        print "Output: \n" + str(self.forward(x_test));
        
nn = NN()

for i in xrange(1000): # trains the NN 1,000 times

  print str(np.mean(np.square(y - nn.forward(X)))) # mean sum squared loss
  nn.train(X,y)   


nn.predict()