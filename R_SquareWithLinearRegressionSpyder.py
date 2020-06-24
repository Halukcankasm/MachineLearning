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


#Visual x(deneyim) / y(maaş)
plt.scatter(x,y,color="blue")
plt.show()


#Visual line(fit line - Linear)
y_head = linear_reg.predict(x)
plt.plot(x,y_head,color="red")

#%% R-Square

from sklearn.metrics import r2_score

print("r_score(hataPayı):",r2_score(y, y_head))