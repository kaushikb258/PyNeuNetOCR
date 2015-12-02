"""
-----------------------------------------
 NEURAL NETWORK WITH BACKPROPAGATION
 AUTHOR: KAUSHIK BALAKRISHNAN, PHD
 kaushikb258@gmail.com
-----------------------------------------
"""

#------------------------------------------------
def set_inputs():

# number of hidden layers
 nhidden = 7

# maximum number of neurons per layer
 max_neurons = 30 

# weight adjustment factor
 beta = 0.5

# number of Neural Net iterations
 niters = 100

# number of inputs/features
 ninputs = 16

# number of outputs/targets
 noutputs = 26

# update procedure
 update_procedure = 1

# update_procedure = 1 for the classical beta approach

# number of neurons per layer
 num_neurons = [ninputs, 18, 25, 20, 24, 28, 20, 22, noutputs] 

 return nhidden, max_neurons, beta, niters, ninputs, noutputs, update_procedure, num_neurons
