import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% create dataset


#class1
x1= np.random.normal(25,5,1000) #Gaussion variable
y1= np.random.normal(25,5,1000) #Gaussion variable
#1000 tane değer üret oratalaması 25 olsun
#25+5=30 , 25-5=20  30 ile 20 arasında değerlerden oluşacak


#class2
x2= np.random.normal(55,5,1000) #Gaussion variable
y2= np.random.normal(65,5,1000) #Gaussion variable


#class3
x3= np.random.normal(55,5,1000) #Gaussion variable
y3= np.random.normal(15,5,1000) #Gaussion variable

x=np.concatenate((x1,x2,x3),axis = 0)
y=np.concatenate((y1,y2,y3),axis = 0)


dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)

# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.scatter(x3,y3)
# plt.show()

# ## %% kmeans algoritması bunları böyle görecek
# plt.scatter(x1,y1,color="black")
# plt.scatter(x2,y2,color="black")
# plt.scatter(x3,y3,color="black")
# plt.show()



#%% dendogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method="ward")
#Clusterin içini normalize -> method="ward"
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

#Dendogramdan en yüksek distance(mesafe)yi kesen dengromram 100 olduğundan 3 cluster oluşturuyoeum

#%% HC
from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage = "ward")
cluster = hiyerartical_cluster.fit_predict(data)

data["label"]=cluster

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color="red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color="blue")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color="green")
plt.show()




























































