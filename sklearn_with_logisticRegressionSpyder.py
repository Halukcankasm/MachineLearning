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



#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train, y_train)
prediction = lr.predict(x_test)
print("test accuracy {}".format(lr.score(x_test,y_test)))
