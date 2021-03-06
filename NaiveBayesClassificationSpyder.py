#%% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%read csv

data=pd.read_csv("CancerData.csv")
print(data.head())


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
"""
data içerisindeki train için 2 tane futuare kaldırıyoum
etkisi olmayan futuareli kaldırıyoruz
axis=1 ile yazılan futuarelerin tüm satırlarını kaldır
inplace=True ile güncelle ve data kaydet
"""

M = data[data.diagnosis == "M"] #KÖTÜ HIYLU TÜMOR

B = data[data.diagnosis == "B"] #İYİ HUYLU TÜMOR

print(M.info()) #212 entrie , 212 tane kötü hıylu tümör var

print(B.info()) #357 entrie , 357 tane kötü hıylu tümör var

#%%scatterplot

plt.scatter(M.radius_mean,M.texture_mean,color="black",label="M(kotu)")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="B(iyi)")
plt.xlabel("radius_mean / yarıçap") # tümörlerin yarı çapı
plt.ylabel("texture_mean / doku ve yapısı") #Tümörerin doku ve yapısı

plt.legend()#label göstermemizi sağlıyor
plt.show()

data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis ]


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%Normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

#%% Naice Bayes

from sklearn.naive_bayes import GaussianNB

nb =GaussianNB()
nb.fit(x_train,y_train)

#%% test

print("print accuracy of nb algo",nb.score(x_test,y_test))












































