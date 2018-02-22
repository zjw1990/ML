#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Feb  5 16:47:10 2018

@author: zhao
"""
# Gradient Descent for y = mx+b
import numpy as np
import matplotlib.pyplot as plt

def compute_err(b,m,data):
    totalErr = 0
    for i in xrange(len(data)):
        dataX = data[i,0]
        dataY = data[i,1]
        #compute residual
        residual = dataY-(m*dataX+b)
        totalErr += residual**2
 
    return totalErr/float(len(data))
        
def compute_gradient(bCurrent,mCurrent,data,learningRate):
    bGradient = 0
    mGradient = 0
    N = float(len(data))
    for i in range(0,len(data)):
        x = data[i,0]
        y = data[i,1]
        bGradient += -(y-(mCurrent*x+bCurrent))*(1/N)
        mGradient += -(y-(mCurrent*x+bCurrent))*x*(1/N)
    newB = bCurrent - learningRate*bGradient
    newM = mCurrent - learningRate*mGradient
    
    return newB,newM
        
def gradient_runner(data,iteration,learning_rate):
    m = 0
    b = 0
    
    for i in range(iteration):
        err = compute_err(b, m, data)
        b,m = compute_gradient(b,m,np.array(data),learning_rate)
        threhold = err - compute_err(b, m, data)
        if threhold < 0.001:
            break
        
       
    return b,m,i

def run():
    #np.random.seed(1)
    #data = np.random.randint(1,100,size=(1000,2))
    data = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0003
    initialB = 0 # initial y-intercept guess
    initialM = 0 # initial slope guess
    num_iterations = 100
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initialB, initialM, compute_err(initialB, initialM, data))
    print 'running'
    [b, m,i] = gradient_runner(data, num_iterations,learning_rate)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(i, b, m, compute_err(b, m, data))
    plot(data,m,b)

    
def plot(data,m,b):
    x = []
    y = []
    for i in data:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y)
    
    x_ = np.linspace(0,100,10)

    y_ = m*x_+b
    plt.plot(x_,y_)
if __name__ == '__main__':
    run()

        
        


