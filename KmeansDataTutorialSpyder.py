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

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()

# ## %% kmeans algoritması bunları böyle görecek
# plt.scatter(x1,y1,color="black")
# plt.scatter(x2,y2,color="black")
# plt.scatter(x3,y3,color="black")
# plt.show()



#%%
from sklearn.cluster import KMeans
wcss=[]

for k  in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()

#%% k = 3 için model

kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color="red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color="blue")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color="green")
plt.show()



































