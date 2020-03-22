# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:39:12 2020

@author: megha
"""
import torch
import numpy as np

def data_cleaning_and_restructuring(test,test_labels,train,train_labels):
        train_filtered_56 = train[((train_labels==2) + (train_labels==5) + (train_labels==7))]
        train_filtered_56_labels = train_labels[((train_labels==2) + (train_labels==5) + (train_labels==7))]
        test_filtered_56 = test[((test_labels==2) + (test_labels==5) + (test_labels==7))]
        test_filtered_56_labels = test_labels[((test_labels==2) + (test_labels==5) + (test_labels==7))]
              
        
        train_filtered_56_reshaped =  train_filtered_56.reshape(-1,28*28)
        test_filtered_56_reshaped = test_filtered_56.reshape(-1,28*28)
        #train_data_56 = np.hstack((np.ones((train_filtered_56_reshaped.shape[0],1)),train_filtered_56_reshaped))
        #test_data_56 = np.hstack((np.ones((test_filtered_56_reshaped.shape[0],1)),test_filtered_56_reshaped))
        
        
        labels = np.zeros(train_filtered_56_labels.shape[0],dtype=int)
        labels[train_filtered_56_labels==2] = 0
        labels[train_filtered_56_labels==5] = 1
        labels[train_filtered_56_labels==7] = 2
        
        test_labels = np.zeros(test_filtered_56_labels.shape[0],dtype=int)
        test_labels[test_filtered_56_labels==2] = 0
        test_labels[test_filtered_56_labels==5] = 1
        test_labels[test_filtered_56_labels==7] = 2
                
        return test_filtered_56_labels,labels,test_labels,train_filtered_56_labels,train_filtered_56,test_filtered_56

def prepare_training_data(train_filtered_56_labels,train_filtered_56,labels):
    train_loader_ip = []
    
    for i in range(train_filtered_56_labels.shape[0]):
        train_loader_ip.append((torch.Tensor(train_filtered_56[i].reshape(1,28,28)),int(labels[i])))

    trainloader = torch.utils.data.DataLoader(train_loader_ip,batch_size=batch_size)
    return trainloader


def prepare_testing_data(test_filtered_56_labels,test_filtered_56,test_labels):
    test_loader_ip = []
    for i in range(test_filtered_56_labels.shape[0]):
      test_loader_ip.append((torch.Tensor(test_filtered_56[i].reshape(1,28,28)),int(test_labels[i])))

    testloader = torch.utils.data.DataLoader(test_loader_ip,batch_size=64)
    return testloader

def train_data(trainloader):
    
    
    net = Net()
    ##define  loss function  and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters())
    
    
    accuracy=0
    correct=0
    #End Your Code
    total=0
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0
        for i,data in enumerate(trainloader,0):

            inputs,labels = data

            optimizer.zero_grad()

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            if i % 100 == 99:    # print every 100 mini-batches
                print('Epoch: %d, Batch: %5d, loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    
                
                

    print('Finished Training')
    return net




def test_data(classes,net,testloader):  
    
    
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    with torch.no_grad():
    
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))