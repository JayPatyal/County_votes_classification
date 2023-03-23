import numpy as np
import math
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average

from numpy.random.mtrand import sample


class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		# print(- 2.0 * (target - prediction))
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			# print(loss)

			grad = self.MSEGrad(pred, yi)
			# print(grad)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		
		self.input = input
		value = np.dot(input, self.w)+self.b
		return value

	def backward(self, gradients):
		#Write backward pass here
		
		w_derivative = np.dot(self.input.T, gradients)
		x_derivative = np.dot(gradients, self.w.T)
		self.w =self.w - self.lr * w_derivative
		self.b= self.b - self.lr * gradients
		return x_derivative


class Sigmoid:

	def __init__(self):
		return None
		

	def forward(self, input):
		#Write forward pass here
		self.sig_val = 1/(1+np.exp(-input))
		
		return self.sig_val
		
	def backward(self, gradients):
		#Write backward pass here
		# print(gradients)
		
		sig_derivative = gradients * (self.sig_val*(1-self.sig_val))
		# print(a)
		return sig_derivative


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=0))

	def train(self, X):
		#training logic here
		#input is array of features (no labels)
		
		self.X = X
		# print(X.shape)
		self.sample, self.features = X.shape
		self.centroids = []
		rand_index = np.random.choice(self.sample, self.k)
		for i in rand_index:
			self.centroids.append(self.X[i])

		for i in range(self.t):
			
			self.clusters = []
			for i in range(self.k):
				self.clusters.append([])
				
			for i, j in enumerate(self.X):
				
				dist = []
				for point in self.centroids:
					dist.append(self.distance(j, point))
				closest_point = np.argmin(dist)
				self.clusters[closest_point].append(i)

			old_centroids = self.centroids
			new_centroids = np.zeros((self.k, self.features))

			for index_val, cluster in enumerate(self.clusters):
				avg_val = average(self.X[cluster])
				new_centroids[index_val] = avg_val
			self.centroids = new_centroids
			distances = []

			for j in range(self.k):
				dist = (self.distance(old_centroids[j], self.centroids[j]))
				distances.append(dist)

			if sum(distances) == 0:
				break
			labels = np.empty(self.sample)
			i =0
			for j in self.clusters:
				i+=1
				for k in j:
					labels[k] = i
		return labels

class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)
		cluster_idx= []
		
		for i in range(len(X)):
			cluster_idx.append(i)
		
		distances = []
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				obj = [self.distance(X[i], X[j]), (i, j)]
				distances.append(obj)
				
		distances = sorted(distances)
		
		cluster_num = len(X)
		while(cluster_num>self.k):
			
			distance, pts = distances.pop()
			pt1 = pts[0]
			pt2 = pts[1]
			
			cluster1_count = cluster_idx.count(cluster_idx[pt1])
			cluster2_count = cluster_idx.count(cluster_idx[pt2])

			if(cluster1_count<cluster2_count):
				
				for i in range(len(cluster_idx)):
					if(cluster_idx[i] == cluster_idx[pt1]):
						cluster_idx[i] = cluster_idx[pt2]
				cluster_num -= 1
			
			else:
				for i in range(len(cluster_idx)):
					if(cluster_idx[i] == cluster_idx[pt2]):
						cluster_idx[i] = cluster_idx[pt1]
				cluster_num -= 1
				
		return cluster_idx
	