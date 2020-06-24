import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("DecionTreeRegressionDaataSet.csv", sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)


#%%decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)
deger = np.array(5.5)
print("5.5 e göre y_head:",tree_reg.predict(deger.reshape(-1,1)))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
#x in minumum değerinden max değerine 0.01 artarak bir array oluştur

y_head=tree_reg.predict(x_)

#%% visualize
plt.scatter(x,y,color="blue")
plt.plot(x_,y_head,color="black")
plt.xlabel("TribünLevel")
plt.xlabel("Ücret")
plt.show()



