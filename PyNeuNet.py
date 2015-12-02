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
import inputdata
from inputdata import *
from userinputs import *
import NeuNet
from NeuNet import *
import sys
import matplotlib.pyplot as plt
import time
#------------------------------------------------

nhidden, max_neurons, beta, niters, ninputs, noutputs, update_procedure, num_neurons = set_inputs()

print "-------------------------------------------"
print "nhidden: ", nhidden
print "max_neurons: ", max_neurons
print "beta: ", beta
print "niters: ", niters
print "ninputs: ", ninputs
print "noutputs: ", noutputs
print "update_procedure: ", update_procedure
print "num_neurons: ", num_neurons
print "-------------------------------------------"

# read raw data
ntrain, ntest, train_in, train_out, test_in, test_out = read_raw_data()

print "ntrain: ", ntrain
print "ntest: ", ntest
sys.stdout.flush()
time.sleep(5)

# weights: w[j,i,k] 
# weight from k-th neuron in level j-1 to i-th neuron in level j

w = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)
wnew = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)
dw = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)
dwold = np.zeros((nhidden+2,max_neurons,max_neurons),dtype=np.float64)

bias = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
biasnew = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
dbias = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
dbiasold = np.zeros((nhidden+2,max_neurons),dtype=np.float64)

act = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
delta = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
error = np.zeros((niters,noutputs),dtype=np.float64)

# initialize weights
print "initializing weights "
sys.stdout.flush()
w, bias = initialize_weights(max_neurons,nhidden,num_neurons)

#---------------------------------------------------------------
print "starting the training "
sys.stdout.flush()
# start the neural net computations (niters iterations)
for it in range(niters):
   
  print "iteration: ", it  
  sys.stdout.flush() 
    
  for ii in range(ntrain):
   act = np.zeros((nhidden+2,max_neurons),dtype=np.float64)   
   act = forward_propagation(max_neurons,nhidden,num_neurons,train_in[ii,:],w,bias)
   
# compute error for each neuron
   delta = np.zeros((nhidden+2,max_neurons),dtype=np.float64)
   delta = compute_error(max_neurons,nhidden,num_neurons,train_out[ii,:],act,w)

# adjust weights
   w, bias = adjust_weights(update_procedure,max_neurons,nhidden,num_neurons,beta,act,w,bias,delta)
   
   for i in range(noutputs):
    error[it,i] = error[it,i] + ( train_out[ii,i] - act[nhidden+1,i] )**2.0

#-------------------------------       
error[:,:] = np.sqrt(error[:,:]/float(ntrain))

# outout error to file

err1 = np.zeros((error.shape[0],error.shape[1]+1),dtype=np.float64)
b = np.array([i for i in range(1,error.shape[0]+1)])
b = b.astype(np.float64)
err1 = np.insert(error, 0, b, axis=1)
np.savetxt('error_out', err1, delimiter=',')

# Plot error
plt.plot(err1[:,0],np.log10(err1[:,1]))
plt.xlabel('iteration #')
plt.ylabel('log10 error')
plt.show()

#-------------------------------
print "-------------------------"

# Apply on test set

correct_pred = 0 

for ii in range(ntest):
   act = np.zeros((nhidden+2,max_neurons),dtype=np.float64)   
   act = forward_propagation(max_neurons,nhidden,num_neurons,test_in[ii,:],w,bias) 
   output = act[nhidden+1,:noutputs]  
   k = np.argmax(output)  
   if (test_out[ii,k]==1):
    correct_pred += 1
     
print "# of correct predictions: ", correct_pred       
print "# on test sets: ", ntest           
               
print "-------------------------"
#-------------------------------