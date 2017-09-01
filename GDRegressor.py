import numpy as np

np.seterr(all='warn')

class GDRegressor:

	learning_rate = 0.001
	max_iter = 1000
	theta = (0,0)
	t0, t1 = 5, 50

	def __init__(self, max_iter = 1000, learning_rate = 0.001):
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def fit(self, X, y):
		self._perform_gradient_descendent(X, y)

	def predict(self, X):
		return X.dot(self.theta)

	def _learning_schedule(self, time):
		return self.t0/ (time + self.t1)


	def _perform_gradient_descendent(self, X, Y):
		self.theta = np.random.rand(X.shape[1], 1)
		size = X.shape[1]
		for epoch in range(self.max_iter):
			gradients = 2/size * X.T.dot(X.dot(self.theta) - Y)
			self.theta = self.theta - self.learning_rate * gradients
