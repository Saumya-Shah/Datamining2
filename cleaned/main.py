import numpy as np
import matplotlib.pyplot as plt
from helping_functions import *
from class_definition import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    

    #READING DATA
    test = np.load("test.npy")
    train = np.load("train.npy")
    test_labels = np.load("test_labels.npy")
    train_labels = np.load("train_labels.npy")
    
    #QUESTION 1 PLOTTING RANDOM IMAGES
    
    #Uncomment the next line to see 10 random images 
    # random_image_display(10,train)
    
    
    # -----------------
    #HYPERPARAMETERS FOR PEGASUS
    regu_para = 0.1
    num_iter = 4000
    batch_size = 100
    # -----------------
    
    #Convert data to model-friendly format
    train_data,train_labels_bin,test_data,test_labels_bin = \
        filter_2_classes(train,train_labels,test,test_labels,2,5)
    #Train using Pegasus
    w,train_accu,test_accu = \
        Pegasus(regu_para,num_iter,batch_size,train_data,train_labels_bin,test_data,test_labels_bin)
    plt.clf()
    plt.plot(train_accu)
    plt.title("train_accuracy vs iterations for Pegasus")
    plt.xlabel("iterations")
    plt.ylabel("training accuracy")
    plt.savefig("results/train_accuracy_pegasus.png")
    
    
    plt.clf()
    plt.plot(test_accu)
    plt.title("test_accuracy vs iterations for Pegasus")
    plt.xlabel("iterations")
    plt.ylabel("testing accuracy")
    plt.savefig("results/test_accuracy_pegasus.png")
    
    train_final_accuracy_Pega = np.sum((np.sum(w*train_data,axis = 1))\
        *train_labels_bin >1)/train_data.shape[0]
    test_final_accuracy_Pega = np.sum((np.sum(w*test_data,axis = 1))\
        *test_labels_bin >1)/test_data.shape[0]
    print("Train accuracy with Pegasus is", train_final_accuracy_Pega*100)
    print("Test accuracy with Pegasus is", test_final_accuracy_Pega*100)
    
    
    # -----------------
    #HYPERPARAMETERS FOR PEGASUS WITH ADAGRAD
    regu_para = 0.1
    num_iter = 200
    batch_size = 100
    # -----------------
    w,train_accu,test_accu = \
        Pega_with_Ada(regu_para,num_iter,batch_size,train_data,train_labels_bin,test_data,test_labels_bin)
    
    plt.clf()
    plt.plot(train_accu)
    plt.title("train_accuracy vs iterations with Adagrad")
    plt.xlabel("iterations")
    plt.ylabel("training accuracy")
    plt.savefig("results/train_accuracy_pega_ada.png")
    
    plt.clf()
    plt.plot(test_accu)
    plt.title("test_accuracy vs iterations with Adagrad")
    plt.xlabel("iterations")
    plt.ylabel("testing accuracy")
    plt.savefig("results/test_accuracy_pega_ada.png")
    
    train_final_accuracy_Ada = np.sum((np.sum(w*train_data,axis = 1))\
        *train_labels_bin >1)/train_data.shape[0]
    test_final_accuracy_Ada = np.sum((np.sum(w*test_data,axis = 1))\
        *test_labels_bin >1)/test_data.shape[0]
    
    print("Train accuracy with Pegasus and Adagrad is", train_final_accuracy_Ada*100)
    print("Test accuracy with Pegasus and Adagrad is", test_final_accuracy_Ada*100)
    
    
    # -----------------
    #HYPERPARAMETERS FOR MULTICLASS ADA
    regu_para = 0.1
    num_iter = 200
    batch_size = 100
    # -----------------
    
    
    train_data_25,train_labels_bin_25,_,_ = \
        filter_2_classes(train,train_labels,test,test_labels,2,5)
    
    train_data_27,train_labels_bin_27,_,_ = \
        filter_2_classes(train,train_labels,test,test_labels,2,7)
    
    train_data_57,train_labels_bin_57,_,_ = \
        filter_2_classes(train,train_labels,test,test_labels,5,7)
    
    train_data_257,train_labels_257,test_data_257,test_labels_257\
        = filter_3_classes(train,train_labels,test,test_labels,2,5,7) 
    
    w25,w27,w57,train_accu,test_accu = Ada_3_classes(regu_para,num_iter,batch_size,train_data_25,train_data_27,train_data_57,train_labels_bin_25,train_labels_bin_27,train_labels_bin_57,train_data_257,train_labels_257,test_data_257,test_labels_257)
    
    plt.clf()
    plt.plot(train_accu)
    plt.title("train_accuracy vs iterations with Multiclass with Adagrad")
    plt.xlabel("iterations")
    plt.ylabel("training accuracy")
    plt.savefig("results/train_accuracy_pega_ada_multi.png")
    
    plt.clf()
    plt.plot(test_accu)
    plt.title("test_accuracy vs iterations with Multiclass with Adagrad")
    plt.xlabel("iterations")
    plt.ylabel("testing accuracy")
    plt.savefig("results/test_accuracy_pega_ada_multi.png")
    
    y_pred_train = multiclass_result(w25,w27,w57,train_data_257)
    y_pred_test = multiclass_result(w25,w27,w57,test_data_257)
    print("Overall Multiclass Train Accuracy",100*(np.sum(y_pred_train==train_labels_257))/train_data_257.shape[0])
    print("Overall Multiclass Test Accuracy",100*(np.sum(y_pred_test==test_labels_257))/test_data_257.shape[0])
    
    
    
    
    
    #========================================================================================
    #QUESTION 6
    
    
    #Hyperparameters for CNN:
    regu_para = 0.1
    num_iter = 3
    batch_size = 128
    epochs=10
    
    
   
    ## restructure data in required format
    
    test_filtered_56_labels,labels,test_labels,train_filtered_56_labels,train_filtered_56,test_filtered_56=data_cleaning_and_restructuring(test,test_labels,train,train_labels)
    
    trainloader=prepare_training_data(train_filtered_56_labels,train_filtered_56,labels,batch_size)
    testloader=prepare_testing_data(test_filtered_56_labels,test_filtered_56,test_labels)
    
    #### Training and testing
    classes = ['two','five','seven']

    net=train_data_func(trainloader)
    test_data_func(classes,net,testloader)
    
