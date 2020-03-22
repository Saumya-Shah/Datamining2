import numpy as np
import matplotlib.pyplot as plt

def random_image_display(num_im,train):
    """Plots num_im images at random from the data set"""
    random_sets = np.random.randint(0,train.shape[0],num_im)
    for i in random_sets:
        plt.imshow(train[i],cmap = "gray")
        plt.show()

def filter_2_classes(train,train_labels,test,test_labels,c1,c2):
    """ Returns train_data, train_binary_labels, test_data, test_binary_labels for classes c1, c2  """
    train_filtered_27 = train[((train_labels==c1) + (train_labels==c2))]
    train_filtered_27_labels = train_labels[((train_labels==c1) + (train_labels==c2))]
    train_filtered_27_labels_binary = 1*(train_filtered_27_labels==c1) -1*(train_filtered_27_labels==c2)  

    test_filtered_27 = test[((test_labels==c1) + (test_labels==c2) )]
    test_filtered_27_labels = test_labels[((test_labels==c1) + (test_labels==c2))]
    test_filtered_27_labels_binary = (test_filtered_27_labels==c1)*1 -1*(test_filtered_27_labels==c2)  

    train_filtered_27_reshaped =  train_filtered_27.reshape(-1,28*28)
    test_filtered_27_reshaped = test_filtered_27.reshape(-1,28*28)
    train_data_27 = np.hstack((np.ones((train_filtered_27_reshaped.shape[0],1)),train_filtered_27_reshaped))
    test_data_27 = np.hstack((np.ones((test_filtered_27_reshaped.shape[0],1)),test_filtered_27_reshaped))
    
    return train_data_27,train_filtered_27_labels_binary,test_data_27,test_filtered_27_labels_binary

def filter_3_classes(train,train_labels,test,test_labels,c1,c2,c3):
    train_filtered_56 = train[((train_labels==c1) + (train_labels==c2) + (train_labels==c3))]
    train_filtered_56_labels = train_labels[((train_labels==c1) + (train_labels==c2) + (train_labels==c3))]
    test_filtered_56 = test[((test_labels==c1) + (test_labels==c2) + (test_labels==c3))]
    test_filtered_56_labels = test_labels[((test_labels==c1) + (test_labels==c2) + (test_labels==c3))]
    
    train_filtered_56_reshaped =  train_filtered_56.reshape(-1,28*28)
    test_filtered_56_reshaped = test_filtered_56.reshape(-1,28*28)
    train_data_56 = np.hstack((np.ones((train_filtered_56_reshaped.shape[0],1)),train_filtered_56_reshaped))
    test_data_56 = np.hstack((np.ones((test_filtered_56_reshaped.shape[0],1)),test_filtered_56_reshaped))

    return train_data_56,train_filtered_56_labels,test_data_56,test_filtered_56_labels

def rectify_w(w,w_norm_limit):
    if np.linalg.norm(w)>w_norm_limit:
        return w*w_norm_limit/np.linalg.norm(w)
    else:
        return w

def Pegasus(regu_para, num_iter, batch_size, train_data_23,train_filtered_23_labels_binary,test_data_23,test_filtered_23_labels_binary):
    w_norm_limit = 1/np.sqrt(regu_para)
    w = np.random.random(train_data_23.shape[1])
    w = rectify_w(w,w_norm_limit)
    train_accu = []
    test_accu = []
    for t in range(num_iter):
        if t%100==0:
            print(t)
        lr = 1/(regu_para*(t+1))
        A_index = np.random.randint(0,train_data_23.shape[0],100)
        A_t = train_data_23[A_index]
        y_t = train_filtered_23_labels_binary[A_index]
        A_t_wrong = A_t[(np.sum(w*A_t,axis = 1))*y_t < 1]
        y_t_wrong = y_t[(np.sum(w*A_t,axis = 1))*y_t < 1]
        #for plotting --------------
        train_accu.append(np.sum((np.sum(w*train_data_23,axis = 1))*train_filtered_23_labels_binary >1)/train_data_23.shape[0])
        test_accu.append(np.sum((np.sum(w*test_data_23,axis = 1))*test_filtered_23_labels_binary >1)/test_data_23.shape[0])
        #for plotting end ----------
        del_t = regu_para*w - (1/batch_size)*np.sum(y_t_wrong.reshape(-1,1)*A_t_wrong,axis = 0)
        w = w-lr*del_t
        w = rectify_w(w,w_norm_limit)
    return w,train_accu,test_accu

def Pega_with_Ada(regu_para, num_iter, batch_size, train_data_23,train_filtered_23_labels_binary,test_data_23,test_filtered_23_labels_binary):
    w_norm_limit = 1/np.sqrt(regu_para)
    w = np.random.random(train_data_23.shape[1])
    w = rectify_w(w,w_norm_limit)
    train_accu = []
    test_accu = []
    G = np.zeros(train_data_23.shape[1])+0.001
    for t in range(num_iter):
        if t%100==0:
            print(t)
        lr = 1/(regu_para*(t+1))
        A_index = np.random.randint(0,train_data_23.shape[0],100)
        A_t = train_data_23[A_index]
        y_t = train_filtered_23_labels_binary[A_index]
        A_t_wrong = A_t[(np.sum(w*A_t,axis = 1))*y_t < 1]
        y_t_wrong = y_t[(np.sum(w*A_t,axis = 1))*y_t < 1]
        #for plotting --------------
        train_accu.append(np.sum((np.sum(w*train_data_23,axis = 1))*train_filtered_23_labels_binary >1)/train_data_23.shape[0])
        test_accu.append(np.sum((np.sum(w*test_data_23,axis = 1))*test_filtered_23_labels_binary >1)/test_data_23.shape[0])
        #for plotting end ----------
        del_t = regu_para*w - (1/batch_size)*np.sum(y_t_wrong.reshape(-1,1)*A_t_wrong,axis = 0)
        w = w-lr*del_t/G**0.5
        G = G + del_t**2
        w = rectify_w(w,w_norm_limit)
    return w,train_accu,test_accu

def multiclass_result(w25,w27,w57,train_data_56):  
    temp = np.sign(np.sum(w25*train_data_56,axis = 1))+10*np.sign(np.sum(w27*train_data_56,axis = 1))+100*np.sign(np.sum(w57*train_data_56,axis = 1))
    temp[(temp==-109)+(temp==-111)] = 7
    temp[(temp==109)+(temp==89)] = 5
    temp[(temp==-89)+(temp==111)] = 2
    return temp
    

def Ada_3_classes(regu_para, num_iter, batch_size,train_data_25,train_data_27,\
    train_data_57,train_filtered_25_labels_binary,train_filtered_27_labels_binary,\
        train_filtered_57_labels_binary,train_data_56,train_filtered_56_labels,\
            test_data_56,test_filtered_56_labels):
    #PEGASUS+ADAGRAD #ONE VS ONE
    w_norm_limit = 1/np.sqrt(regu_para)
    w25 = np.random.random(train_data_25.shape[1])
    w25 = rectify_w(w25,w_norm_limit)
    w27 = np.random.random(train_data_27.shape[1])
    w27 = rectify_w(w27,w_norm_limit)
    w57 = np.random.random(train_data_57.shape[1])
    w57 = rectify_w(w57,w_norm_limit)

    train_accu = []
    test_accu = []
    G25 = np.zeros(train_data_25.shape[1])+0.001
    G27 = np.zeros(train_data_27.shape[1])+0.001
    G57 = np.zeros(train_data_57.shape[1])+0.001
    for t in range(num_iter):
        if t%100==0:
            print(t)
        lr = 1/(regu_para*(t+1))
        A25_index = np.random.randint(0,train_data_25.shape[0],100)
        A27_index = np.random.randint(0,train_data_27.shape[0],100)
        A57_index = np.random.randint(0,train_data_57.shape[0],100)

        A25_t = train_data_25[A25_index]
        A27_t = train_data_27[A27_index]
        A57_t = train_data_57[A57_index]

        y25_t = train_filtered_25_labels_binary[A25_index]
        y27_t = train_filtered_27_labels_binary[A27_index]
        y57_t = train_filtered_57_labels_binary[A57_index]


        A25_t_wrong = A25_t[(np.sum(w25*A25_t,axis = 1))*y25_t < 1]
        y25_t_wrong = y25_t[(np.sum(w25*A25_t,axis = 1))*y25_t < 1]
        A27_t_wrong = A27_t[(np.sum(w27*A27_t,axis = 1))*y27_t < 1]
        y27_t_wrong = y27_t[(np.sum(w27*A27_t,axis = 1))*y27_t < 1]
        A57_t_wrong = A57_t[(np.sum(w57*A57_t,axis = 1))*y57_t < 1]
        y57_t_wrong = y57_t[(np.sum(w57*A57_t,axis = 1))*y57_t < 1]

        #for plotting --------------
        y_pred_train = multiclass_result(w25,w27,w57,train_data_56)
        y_pred_test = multiclass_result(w25,w27,w57,test_data_56)
        train_accu.append((np.sum(y_pred_train==train_filtered_56_labels))/train_data_56.shape[0])
        test_accu.append((np.sum(y_pred_test==test_filtered_56_labels))/test_data_56.shape[0])
        #for plotting end ----------

        del25_t = regu_para*w25 - (1/batch_size)*np.sum(y25_t_wrong.reshape(-1,1)*A25_t_wrong,axis = 0)
        del27_t = regu_para*w27 - (1/batch_size)*np.sum(y27_t_wrong.reshape(-1,1)*A27_t_wrong,axis = 0)
        del57_t = regu_para*w57 - (1/batch_size)*np.sum(y57_t_wrong.reshape(-1,1)*A57_t_wrong,axis = 0)

        w25 = w25-lr*del25_t/G25**0.5
        w27 = w27-lr*del27_t/G27**0.5
        w57 = w57-lr*del57_t/G57**0.5

        G25 = G25 + del25_t**2
        G27 = G27 + del27_t**2
        G57 = G57 + del57_t**2
        
        w25 = rectify_w(w25,w_norm_limit)
        w27 = rectify_w(w27,w_norm_limit)
        w57 = rectify_w(w57,w_norm_limit)

    return w25,w27,w57,train_accu,test_accu