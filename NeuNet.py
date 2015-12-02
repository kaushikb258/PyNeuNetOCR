"""
-----------------------------------------
 NEURAL NETWORK WITH BACKPROPAGATION
 AUTHOR: KAUSHIK BALAKRISHNAN, PHD
 kaushikb258@gmail.com
-----------------------------------------
"""
import numpy as np
import pandas as pd
import random
import sys

#-------------------------------
# Randomly assign weights from -1 to +1

def initialize_weights(max_neurons,nhidden,num_neurons):
    
    w = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)
    bias = np.zeros((nhidden+2,max_neurons),dtype=np.float64)

    for j in range(1,nhidden+2):
     for i in range(num_neurons[j]):
      for k in range(num_neurons[j-1]):
       r = np.random.rand(1)    
       r = -1.0 + 2.0*r 
       w[j,i,k] = r

    for j in range(1,nhidden+2):
     for i in range(num_neurons[j]):
       r = np.random.rand(1)    
       r = -1.0 + 2.0*r   
       bias[j,i] = r      

    return w, bias
     
#-------------------------------
# Forward propagation for one input set

def forward_propagation(max_neurons,nhidden,num_neurons,d_in,w,bias):
   
# Note that d_in is a 1D array here

    if(d_in.shape[0] > max_neurons):
       sys.exit("max_neurons must be larger than number of input features! ") 

    act = np.zeros((nhidden+2,max_neurons),dtype=np.float64)

    for j in range(d_in.shape[0]):    
     act[0,j] = d_in[j]

# level j
# k-th neuron in level j-1 to i-th neuron in level j

    for j in range(1,nhidden+2):
     for i in range(num_neurons[j]):
        x = bias[j,i]
        for k in range (num_neurons[j-1]):
          x = x + act[j-1,k]*w[j,i,k]      
        act[j,i] = sigmoid(x) 

    return act

#-------------------------------
# Sigmoid function

def sigmoid(x):
  s = 1.0/(1.0 + np.exp(-x))
  return s

#-------------------------------
# Compute error in each neuron

def compute_error(max_neurons,nhidden,num_neurons,target_out,act,w):

   delta = np.zeros((nhidden+2,max_neurons),dtype=np.float64)

# note that by default delta = 0 for input units

# output (last) layer
   j = nhidden+1
   for i in range(num_neurons[j]): 
    delta[j,i] = act[j,i]*(1.0-act[j,i])*(target_out[i]-act[j,i])

# hidden layers
   for j in range(nhidden,0,-1): 
    for i in range(num_neurons[j]): 
     x = 0.0
     for k in range(num_neurons[j+1]):
      x = x + delta[j+1,k]*w[j+1,k,i]
     delta[j,i] = act[j,i]*(1.0-act[j,i])*x

   return delta

#----------------------------------------------
# Adjust weights

def adjust_weights(update_procedure,max_neurons,nhidden,num_neurons,beta,act,w,bias,delta):

   if(update_procedure==1):
    wnew = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)
    biasnew = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
    for j in range(1,nhidden+2):
     for k in range(num_neurons[j-1]):
      for i in range(num_neurons[j]):
	wnew[j,i,k] = w[j,i,k] + beta*delta[j,i]*act[j-1,k]
    
    for j in range(1,nhidden+2):
     for i in range(num_neurons[j]):
      biasnew[j,i] = bias[j,i] + beta*delta[j,i]

   else:
    print "ERROR IN update_procedure"
    
    
   return wnew, biasnew  

#----------------------------------------------