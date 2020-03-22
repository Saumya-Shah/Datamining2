# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:20:06 2020

@author: megha
"""
from class_definition import *
import numpy as np
import matplotlib.pyplot as plt


        
if __name__ == '__main__':
    
  
    #Hyperparameters:
    regu_para = 0.1
    num_iter = 3
    batch_size = 128
    epochs=10
    
    
    # get_data
    test = np.load("test.npy")
    test_labels = np.load("test_labels.npy")
    train = np.load("train.npy")
    train_labels = np.load("train_labels.npy")
    
    ## restructure data in required format
    
    test_filtered_56_labels,labels,test_labels,train_filtered_56_labels,train_filtered_56,test_filtered_56=data_cleaning_and_restructuring(test,test_labels,train,train_labels)
    
    trainloader=prepare_training_data(train_filtered_56_labels,train_filtered_56,labels)
    testloader=prepare_testing_data(test_filtered_56_labels,test_filtered_56,test_labels)
    
    #### Training and testing
    classes = ['two','five','seven']

    net=train_data(trainloader)
    test_data(classes,net,testloader)
    
    
    
        