"""
-----------------------------------------
 NEURAL NETWORK WITH BACKPROPAGATION
 AUTHOR: KAUSHIK BALAKRISHNAN, PHD
 kaushikb258@gmail.com
-----------------------------------------
"""

import pandas as pd
import numpy as np
import sys
import random

def read_raw_data():
     
     
     path = '/home/kaushik/Canopy/kaushik_py/PyNeuNet_OCR/letterdata.csv'
     d = pd.read_csv(path)     
     d = np.array(d)
     
     train_in = []
     test_in = []
     train_out = []
     test_out = []
    
     nrow = d.shape[0]
     ncol = d.shape[1]
    
     for i in range(nrow):
       r = np.random.rand(1)
       t = ord(d[i,0].lower()) - 96
       q = np.zeros((26),dtype=np.int)
       q[t-1] = 1
       if(r<=0.8):
         train_in.append(d[i,1:ncol])   
         train_out.append(q)
       else:
         test_in.append(d[i,1:ncol])
         test_out.append(q)   
             
     train_in = np.array(train_in)
     test_in = np.array(test_in)
     train_out = np.array(train_out)
     test_out = np.array(test_out)                      
                                                                              
     ntrain = train_in.shape[0]
     ntest = test_in.shape[0]
            
     return ntrain, ntest, train_in, train_out, test_in, test_out   
#---------------------------------     