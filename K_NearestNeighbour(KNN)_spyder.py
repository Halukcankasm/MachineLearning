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

# plt.scatter(M.radius_mean,M.texture_mean,color="black",label="M(kotu)")
# plt.scatter(B.radius_mean,B.texture_mean,color="green",label="B(iyi)")
# plt.xlabel("radius_mean / yarıçap") # tümörlerin yarı çapı
# plt.ylabel("texture_mean / doku ve yapısı") #Tümörerin doku ve yapısı

# plt.legend()#label göstermemizi sağlıyor
# plt.show()

data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis ]


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%Normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

#%% KNN model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)# n_neighbors = K
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print("{} Knn score {}".format(3,knn.score(x_test,y_test)))

#%% find optimizal K value
score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))#score değerileri listede depolanmış oluyor
    
    

plt.plot(range(1,15),score_list)
plt.xlabel("K values")
plt.ylabel("acuuracy")
plt.show()

#K=8 değerine sahip ise test sonuçları en iyi değeri veriyor











































