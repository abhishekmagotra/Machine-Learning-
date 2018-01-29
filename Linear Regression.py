import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:1].values
X
Y = dataset.iloc[:,1]
Y

from sklearn.cross_validation import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_train,Y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Sal vs Exp (Train)')
plt.xlabel('Expereince(Train)')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,Y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Exp vs Sal (Test)')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()