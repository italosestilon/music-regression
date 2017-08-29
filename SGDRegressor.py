import numpy as np

np.seterr(all='warn')

class SGDRegressor:

	learning_rate = 0.001
	max_iter = 1000
	theta = (0,0)
	t0, t1 = 5, 50
	history = []

	def __init__(self, max_iter = 1000, learning_rate = 0.001):
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def fit(self, X, y):
		self._perform_gradient_descendent(X, y)

	def getErrorHistory(self):
		return self.history

	def predict(self, X):
		return X.dot(self.theta)

	def _learning_schedule(self, time):
		return self.t0/ (time + self.t1)


	def _perform_gradient_descendent(self, X, Y):

		print("shape of X ", X.shape)

		size = X.shape[1]
		self.theta = np.random.rand(X.shape[1], 1)
		self.history = np.arrange(size)

		#print("shape of theta", self.theta)
		for epoch in range(self.max_iter):

			#print("Epoch", epoch)

			for i in range(size):

				j = np.random.randint(X.shape[0])
				x_i = X[j:j+1,:]
				y_i = Y[j:j+1]

				error = x_i.dot(self.theta) - y_i

				gradients = 2*x_i.T.dot(error)

				self.theta = self.theta - self.learning_rate * gradients

				self.history = np.append(self.history, [error])

