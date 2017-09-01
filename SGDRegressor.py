import numpy as np
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.metrics import mean_squared_error

class SGDRegressor:

	learning_rate = 0.001
	max_iter = 1000
	theta = (0,0)
	history = []
	train_erros = []
	batch_size = 100

	def __init__(self, max_iter = 200, learning_rate = 0.001, batch_size = 50, penalty=None):
	    self.learning_rate = learning_rate
	    self.max_iter = max_iter
	    self.batch_size = batch_size
	    self.model = SGD(penalty=penalty, learning_rate='optimal', max_iter = 1, eta0=learning_rate)

	def fit(self, X, y):
		self._perform_gradient_descendent_with_sklearn(X, y)

	def getTrain_errors(self):
	    return self.train_erros

	def getErrorHistory(self):
	    return self.history

	def predict(self, X):
	    return self.model.predict(X)

	def _get_batches(self, X, Y, batch_size):
	    
	    limit = X.shape[0] - batch_size + 1

	    size = int(X.shape[0]/batch_size)

	    starting = np.random.randint(limit, size=size)
	    for i in starting:
	        batch_x = X[i:i+batch_size,:]
	        batch_y = Y[i:i+batch_size]

	        yield batch_x, batch_y

	                
	def score(self, X, Y):
	    return self.model.score(X, Y)


	def _perform_gradient_descendent_with_sklearn(self, X, Y,X_test_scaled=None, Y_test=None):     
		size = X.shape[1]

		self.history = np.arange(size)

		for epoch in range(self.max_iter):
			indices = np.arange(X.shape[0])
			np.random.shuffle(indices)

			batchs = self._get_batches(X[indices,:], Y[indices], self.batch_size)

			for batch_x, batch_y in batchs:
				self.model.partial_fit(batch_x, batch_y)

			y_train_predict = self.model.predict(X)
			self.train_erros.append(np.sqrt(mean_squared_error(Y, y_train_predict)))