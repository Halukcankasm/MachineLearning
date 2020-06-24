#%% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%read csv

data=pd.read_csv("CancerData.csv")
print("Data info",data.info())
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
"""
data içerisindeki train için 2 tane futuare kaldırıyoum
etkisi olmayan futuareli kaldırıyoruz
axis=1 ile yazılan futuarelerin tüm satırlarını kaldır
inplace=True ile güncelle ve data kaydet
"""
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


#%% normalization

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#Train için futuareliri 1-0 arasında bir değer vererek train işlemini optimize ettik.
#Futuareler arası üstünlük sağlamaması
#Normalize fun = (x-min(x))/(max(x-min(x)))

#%% train - test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#test_size=0.2 => x ve y nin %80 train ,%20 test olarak böl

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

print("x_train",x_train.shape)
print("y_train",y_train.shape)
print("x_test",x_test.shape)
print("y_test",y_test.shape)

#%% parameter in initialize and sigmoid function
#dimension = 30 ,30 tane futuaremiz var
def initialize_weights_and_bias(dimentions):
    w=np.full((dimentions,1),0.01)
    b=0.0
    return w,b
#w,b = initialize_weightn_and_bias(30)


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#%%
    
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    """
    x_train,(30,455) olan futuareyi , matrix w ile çarpmam lazım
    w =(30,1)
    """
    z = np.dot(w.T,x_train) + b # w*x +b , ağırlıklar*futuareler + bias
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # x_train.shape[1] = 455 yani /455 yaparak normalize ediyoruz
    
    # backward propagation
    #w,b türebini alarak update ,derivative_weight = w'in türevi
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0 #Kötü huylu tümör
        else:
            Y_prediction[0,i] = 1 #İyi huylu tümör

    return Y_prediction

# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 30)
#logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 3)    
# test accuracy: 78

   
#logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 20)    
# test accuracy: 94 , cost=0.5

#logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 30)
# Cost after iteration 20: 0.266641
# test accuracy: 95.6140350877193 %








































































