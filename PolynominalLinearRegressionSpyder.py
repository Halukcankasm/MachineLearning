import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("PolynominalRegression.csv",sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

#%% LinearRegression
LR = LinearRegression()

LR.fit(x,y) #x,y değerlerine göre en uygun line uyguluyoruz , fit a line


#%% predict
y_head =LR.predict(x)

deger =np.array(10000)
plt.plot(x,y_head,color="red",label="linear")
print("10.000 tl arabanın hızı",LR.predict(deger.reshape(-1,1)))


#%%polynomial regression => y = b0 + b1*x + b2*x^2 + b3*x^3 ....+bn*x^n 

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)
#degree = derece , n =degree

x_polynomial = polynomial_regression.fit_transform(x)
#x(araba fiyatı) i 2.dereceden polynomial futuareye çevir

LR_2 = LinearRegression()
LR_2.fit(x_polynomial,y )

#%%

y_head_2=LR_2.predict(x_polynomial)

plt.plot(x,y_head_2,color="black",label="polynomial")
plt.legend()
plt.show()