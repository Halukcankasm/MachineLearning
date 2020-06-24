import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("DecionTreeRegressionDaataSet.csv", sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators = 100,random_state=42)
#n_estimators = kaç tane tree yazacağım

RF.fit(x,y)

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=RF.predict(x_)


plt.scatter(x, y, color="blue")
plt.plot(x_,y_head,color="black")
plt.xlabel("TribünLevel")
plt.xlabel("Ücret")
plt.show()


#DecionTree'den farkı deciontree 1 tane tree , RandomForestReg 100 tane tree kul.