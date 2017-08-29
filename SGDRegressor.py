import numpy as np

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

	def fit(self, X, y):
		self._perform_gradient_descendent(X, y)

	def getErrorHistory(self):
		return self.history

	def predict(self, X):
		return X.dot(self.theta)

	def _learning_schedule(self, time):
		return self.t0/ (time + self.t1)

	def _get_batches(self, X, Y, batch_size):
		size = X.shape[0] - batch_size + 1
		
		starting = np.random.randint(size, size=size)
		for i in starting:
			batch_x = X[i:i+batch_size,:]
			batch_y = Y[i:i+batch_size]

			yield batch_x, batch_y



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
				gradients = (2/self.batch_size)*batch_x.T.dot(error)
				self.theta = self.theta - self.learning_rate * gradients
