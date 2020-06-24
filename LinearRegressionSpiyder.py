#import library
import pandas as pd 
import matplotlib.pyplot as plt

#import data
data = pd.read_csv("linear_regression_dataset.csv",sep = ";")


#plot data
plt.scatter(data.deneyim,data.maas)
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.show()


#%% Linear Regression

#sklearn Library
from sklearn.linear_model import LinearRegression

#Linear regression model
linear_reg = LinearRegression()

x =data.deneyim.values.reshape(-1,1) #değerlerini aldık, (14,) = (14,1)
y =data.maas.values.reshape(-1,1) #değerlerini aldık, (14,) = (14,1)


#fit line ,Linear regression fit 
linear_reg.fit(x,y)

#%%Prediction

import numpy as np

linear_reg.predict([[0]])
#linear_reg.predict([[0]])

#Line y ekseninde kestiği nokta
b0=linear_reg.predict([[0]])
print("b0:",b0)#y eksenini kestiği nokta , intercept

b0=linear_reg.intercept_
print("b0:",b0)

b1=linear_reg.coef_
print("b1",b1) #êğim slope

#y(maaş) = 1663+1138*x(deneyim)

linear_reg.predict([[11]]) #11 yıllık tecrübedeki maaş


#Visual x(deneyim) / y(maaş)
plt.scatter(x,y,color="blue")
plt.show()


#Visual line(fit line - Linear)
y_head = linear_reg.predict(x)
plt.plot(x,y_head,color="red")









