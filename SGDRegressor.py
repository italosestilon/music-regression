import numpy as np
from sklearn.linear_model import SGDRegressor as SGD

np.seterr(all='warn')

class SGDRegressor:

	learning_rate = 0.001
	max_iter = 1000
	theta = (0,0)
	t0, t1 = 5, 50
	history = []
	batch_size = 100

	def __init__(self, max_iter = 1000, learning_rate = 0.001, batch_size = 100):
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.model = SGD(penalty=None, learning_rate='optimal', max_iter = 1)

	def fit(self, X, y):
		#self._perform_gradient_descendent(X, y)
		self._perform_gradient_descendent_with_sklearn(X, y)

	def getErrorHistory(self):
		return self.history

	def predict(self, X):
		return self.model.predict(X)

	def _learning_schedule(self, time):
		return self.t0/ (time + self.t1)

	def _get_batches(self, X, Y, batch_size):
		
		size = X.shape[0] - batch_size + 1
		
		starting = np.random.randint(size, size=size)
		for i in starting:
			batch_x = X[i:i+batch_size,:]
			batch_y = Y[i:i+batch_size]

			yield batch_x, batch_y

	def score(self, X, Y):
		return self.model.score(X, Y)


	def _perform_gradient_descendent_with_sklearn(self, X, Y):
		#print("shape of Y", Y.shape)
		size = X.shape[1]
		#self.theta = np.random.rand(X.shape[1], 1)
		self.history = np.arange(size)

		#max_iter is just to avoid the warning message but it will not be used.

		for epoch in range(self.max_iter):

			#print("epocha", epoch)
			batchs = self._get_batches(X, Y, self.batch_size)

			for batch_x, batch_y in batchs:
				self.model.partial_fit(batch_x, batch_y)

			#print("score", self.model.score(X, Y))

	
	def _perform_gradient_descendent(self, X, Y):
		print("shape of Y", Y.shape)
		size = X.shape[1]
		self.theta = np.random.rand(X.shape[1], 1)
		self.history = np.arange(size)

		for epoch in range(self.max_iter):

			print("epocha", epoch)
	
			indices = np.arange(X.shape[0])
			np.random.shuffle(indices)
			batchs = self._get_batches(X[indices,:], Y[indices], self.batch_size)


			for batch_x, batch_y in batchs:

				batch_y = batch_y.reshape(self.batch_size, 1)
				error = np.subtract(batch_x.dot(self.theta), batch_y)

				print("Error", error)

				gradients = (2/self.batch_size)*batch_x.T.dot(error)
				self.theta = self.theta - self.learning_rate * gradients

