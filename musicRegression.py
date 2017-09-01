import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_regression

from SGDRegressor import SGDRegressor

from SGDRegressor import SGDRegressor

def load_data(path):
    #return np.loadtxt(path, delimiter=',')
    return pd.read_csv(path, header=None)


data = load_data('year-prediction-msd-train.txt')

#data=pd.read_csv('year-prediction-msd-train.txt',header=None)

data  = np.asarray(data)

Y = data[:,0]
X = data[:,1:]

X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#feature selecting

#scaling the data

X_scaled = StandardScaler().fit_transform(X_training)
X_test_scaled = StandardScaler().fit_transform(X_test)

regression = linear_model.LinearRegression()

regression.fit(X_scaled, Y_training)

y_pred = regression.predict(X_test_scaled)

print("Error : %.2f" %mean_squared_error(Y_test, y_pred))

#Linear regression with stochastic gradient descent

sgd = SGDRegressor(learning_rate=0.001, max_iter=50, batch_size=50)
sgd.fit(X_scaled, Y_training)


history = sgd.getErrorHistory()

sgd_y_pred = sgd.predict(X_test_scaled)

print(sgd_y_pred)
print("Error os SGD : ", mean_squared_error(Y_test, sgd_y_pred))
