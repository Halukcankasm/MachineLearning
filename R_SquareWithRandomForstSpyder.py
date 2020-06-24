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

y_head=RF.predict(x)

#%% R-Square

from sklearn.metrics import r2_score

print("r_score(hataPayı):",r2_score(y, y_head))