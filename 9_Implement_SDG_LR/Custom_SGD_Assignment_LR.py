import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialize_weights(row_vector):
    ''' In this function, we will initialize our weights and bias'''
    w = np.zeros(row_vector.shape)
    b = 0
    return w,b

def sigmoid(z):
    ''' In this function, we will return sigmoid of z'''
    return 1/(1+np.exp(-z)) 

def logloss(y_true,y_pred):
    # you have been given two arrays y_true and y_pred and you have to calculate the logloss
    #while dealing with numpy arrays you can use vectorized operations for quicker calculations as compared to using loops
    #https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html
    #https://www.geeksforgeeks.org/vectorized-operations-in-numpy/
    #write your code here
    loss = -1*np.sum(np.log10(y_pred)*y_true + (1-y_true)*np.log10(1-y_pred))/len(y_true)
    return loss

def gradient_dw(x,y,w,b,alpha,N):
    '''In this function, we will compute the gardient w.r.to w '''
    dw = x*(y - sigmoid(np.dot(w,x)+b)) - alpha*w/N
    return dw

def gradient_db(x,y,w,b):
    '''In this function, we will compute gradient w.r.to b '''
    db = (y - sigmoid(np.dot(w,x)+b))
    return db

def pred(w,b, X):
    N = len(X)
    predict = []
    for i in range(N):
        z=np.dot(w,X[i])+b
        predict.append(sigmoid(z))
    return np.array(predict)
def train(X_train,y_train,X_test,y_test,epochs,alpha,eta0):
    ''' In this function, we will learn weights and biases of logistic regression'''
    
    w,b = initialize_weights(X_train[0])    
    train_loss = []
    test_loss = []
    w,b = initialize_weights(X_train[0]) 
    
    for i in tqdm(range(epochs)):
      
        for i in range(X_train.shape[0]):# No rows is shape 0 
            #compute gradient w.r.to w (call the gradient_dw() function)
            dw = gradient_dw(X_train[i],y_train[i],w,b,alpha,X_train.shape[1])
            #compute gradient w.r.to b (call the gradient_db() function)
            db = gradient_db(X_train[i],y_train[i],w,b)
            #update w, b
            w += dw*eta0
            b += db*eta0
        #After each epoch collect train and test error 
        pred_train = pred(w,b,X_train)
        pred_test  = pred(w,b,X_test) 
        train_step_loss = logloss(y_train,pred_train)
        test_step_loss = logloss(y_test,pred_test)
        
        train_loss.append(train_step_loss)
        test_loss.append(test_step_loss)

    return w,b,train_loss,test_loss


if __name__ =='__main__':

    X, y = make_classification(n_samples=50000, n_features=15, n_informative=10, n_redundant=5,n_classes=2, weights=[0.7], class_sep=0.7, random_state=15)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    dim=X_train[0] 
    
    w,b = initialize_weights(dim)
    
    print("Weights initialized")
    print('w =',(w))
    print('b =',str(b))

    alpha=0.001
    eta0=0.001
    N=len(X_train)
    epochs=2000
    w,b,train_loss,test_loss=train(X_train,y_train,X_test,y_test,epochs,alpha,eta0)
    plt.plot([i for i in range(len(train_loss))],train_loss,color='blue',label="train error")
    plt.plot([i for i in range(len(train_loss))],test_loss,color='red',label="test error")
    plt.xlabel("Number of epochs ->")
    plt.ylabel("log-loss ->")
    plt.legend()
    plt.savefig("Error_Plot.jpeg")