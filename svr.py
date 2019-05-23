#SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the datset
dataset = pd.read_csv('Position_Salaries.csv')
X =  dataset.iloc[:,1:2].values
Y =  dataset.iloc[:, 2].values

#Splitting the dataset into Training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)"""

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
Y = Y.reshape(-1,1) 
X = sc_X.fit_transform(X) 
Y = sc_Y.fit_transform(Y)

#Fitting the regerssion model to the data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)


#Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[5]]))))

#Visualising the Regression Results
plt.scatter(X,Y, color= 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Regression Results(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color= 'red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (SVR model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

