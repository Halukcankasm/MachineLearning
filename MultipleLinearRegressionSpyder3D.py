from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



df=pd.read_csv("multiple_linear_regression_dataset.csv",sep =";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)


multiple_linear_regression = LinearRegression()

#x ve y kullanarak line fit et
multiple_linear_regression.fit(x,y)

print("b0:",multiple_linear_regression.intercept_)
print("b1:",multiple_linear_regression.coef_)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_duzlem =df.maas.values.reshape(-1,1)
x_duzlem =df.deneyim.values.reshape(-1,1)
z_duzlem =df.yas.values.reshape(-1,1)



ax.scatter(x_duzlem, y_duzlem, z_duzlem, c='r', marker='o')

ax.set_xlabel('deneyim')
ax.set_ylabel('Maaş')
ax.set_zlabel('yaş')

plt.show()


y_head = multiple_linear_regression.predict(x)

ax.plot_wireframe(x_duzlem,y_head,z_duzlem)
plt.show()