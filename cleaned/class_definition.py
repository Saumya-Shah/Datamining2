# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:16:59 2020

@author: megha
"""

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim


class Net(nn.Module):
  def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5, stride=1, padding=2)   
        #TODO : Design your network, you are allowed to explore your own architecture
        #       But you should achieve a better overall accuracy than the baseline network.
        #       Also, if you do design your own network, include an explanation 
        #       for your choice of network and how it may be better than the 
        #       baseline network in your latex.
        
        #Begin Your Code
        
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.mxpool1 = torch.nn.MaxPool2d(kernel_size=(2,2),stride=1,padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3), padding=(1,1))
        self.relu2 = torch.nn.ReLU()
        self.mxpool2 = torch.nn.MaxPool2d(kernel_size=(2,2),stride=1,padding=0)
        self.fc=torch.nn.Linear(10816 ,64)
        self.relu3 = torch.nn.ReLU()
        self.fc2=torch.nn.Linear(64, 3)
        self.smax=nn.Softmax()


  def forward(self, x):

    x= self.conv1(x)      
    x = self.relu1(x)   
    x = self.mxpool1(x)
    # print(x.shape)
    x = self.conv2(x)
    # print(x.shape)
    x = self.relu2(x)
    x = self.mxpool2(x)
    # print(x.shape)
    x=x.reshape(x.size(0),-1)
    x=self.fc(x)
    x = self.relu3(x)
  
    x= self.fc2(x)

    return x



