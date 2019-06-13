# Optimization: MNIST
# Hongzheng Chen 17341015

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

MNIST_PATH = "./mnist/mnist.npz"

M_train = 60000
M_test = 10000
batch_size = M_train
n = 28 * 28
C = 10
MAX_ITER = 2000

data = np.load(MNIST_PATH)
X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]
X_train.resize((M_train,n))
y_train.resize((M_train,1))
X_test.resize((M_test,n))
y_test.resize((M_test,1))
X_train = np.column_stack((X_train,np.ones(M_train)))
X_test = np.column_stack((X_test,np.ones(M_test)))

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

def normalize(Xin):
	Xin = Xin.astype(np.float64)
	Xin -= np.mean(Xin, axis=1, keepdims=True)
	Xin /= np.std(Xin, axis=1, keepdims=True)
	return Xin

X_train = normalize(X_train)
X_test = normalize(X_test)

class SoftmaxRegression():

	def train(self, X, y_true, n_classes, n_iters=10, learning_rate=0.1, batch_size=1):
		self.n_samples, n_features = X.shape # (M, C)
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.weights = np.random.rand(self.n_classes, n_features)
		all_losses = []
		all_accuracy = []
		all_weights = []
		
		for i in range(n_iters):
			batch_index = np.array(random.sample(range(self.n_samples),self.batch_size))
			X_batch, y_batch = X[batch_index], y_true[batch_index]
			scores = self.compute_scores(X_batch) # w^T.x
			probs = self.softmax(scores)
			y_one_hot = self.one_hot(y_batch)

			loss = self.nll_loss(y_one_hot, probs) # target function
			all_losses.append(loss)

			# gradient descent -> update weights
			old_weights = self.weights.copy()
			dw = (1 / self.batch_size) * np.dot(X_batch.T, (probs - y_one_hot))
			self.weights = self.weights - learning_rate * dw.T
			all_weights.append(self.weights)

			if i % 100 == 0 or i == n_iters - 1:
				y_predict = self.predict(X_test)
				all_accuracy.append((np.sum(y_predict == y_test) / X_test.shape[0]) * 100)
				print(f'Iteration number: {i}, loss: {np.round(loss, 4)}, accuracy: {all_accuracy[-1]}%')

		return all_weights, all_losses, all_accuracy

	def predict(self, X):
		scores = self.compute_scores(X)
		probs = self.softmax(scores)
		return np.argmax(probs, axis=1)[:, np.newaxis]

	def softmax(self, scores):
		exp = np.exp(scores) # (n_samples, n_classes)
		sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True) # sum along classes
		softmax = exp / sum_exp
		return softmax

	def compute_scores(self, X):
		# X: (n_samples, n_features)
		# scores: (n_samples, n_classes)
		return np.dot(X, self.weights.T)

	def nll_loss(self, y_true, probs):
		loss = - (1 / self.batch_size) * np.sum(y_true * np.log(probs))
		return loss

	def one_hot(self, y):
		one_hot = np.zeros((y.size, self.n_classes))
		one_hot[np.arange(y.size), y.T] = 1
		return one_hot

def plot_res(weights,loss,accuracy):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	lns1 = ax.plot(loss, label="NLLLoss")
	ax.set_xlabel("Number of iterations (k)")
	ax.set_ylabel("Loss")
	ax2 = ax.twinx() # two y-axes
	x = np.arange(0,len(accuracy)*100,100)
	lns2 = ax2.plot(x,accuracy, "r", label="Accuracy")
	ax2.set_ylabel("Accuracy")
	yticks = mtick.FormatStrFormatter("%d%%")
	ax2.yaxis.set_major_formatter(yticks)

	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax.legend(lns,labs,loc=0)

	plt.show()

	plt.cla()
	opt_dist = np.zeros(len(weights))
	for i in range(len(weights)):
		opt_dist[i] = np.linalg.norm(weights[i] - weights[-1])
	plt.plot(opt_dist, label='$||W^{(k)}-W^\\star||_F$')
	plt.xlabel("Number of iterations (k)")
	plt.ylabel("Distance")
	plt.legend(loc=0)
	plt.show()

def plot_batch_size(accuracy,string=""):
	x = np.arange(0,len(accuracy)*100,100)
	plt.plot(x,accuracy,label="bs={}".format(string))

def train(max_iter,batch_size):
	time_start = time.time()
	regressor = SoftmaxRegression()
	weights, loss, accuracy = regressor.train(X_train, y_train, learning_rate=0.1, n_iters=max_iter, n_classes=C, batch_size=batch_size)
	time_used = time.time() - time_start
	print("Training time: {}s".format(time_used))
	return weights, loss, accuracy, time_used

def train_batch_size():
	plt.cla()
	for size, max_iter in [(1,2500),(10,2500),(100,2500),(1000,2500),(10000,2500),(60000,2500)]:
		weights, loss, accuracy, time_used = train(max_iter,size)
		plot_batch_size(accuracy,str(size))
	plt.xlabel("Number of iterations (k)")
	plt.ylabel("Accuracy")
	plt.legend(loc=0)
	plt.show()

if __name__ == '__main__':

	# Gradient descent
	weights, loss, accuracy, time_used = train(2000,60000)
	plot_res(weights,loss,accuracy)

	# SGD
	weights, loss, accuracy, time_used = train(60000,1)
	plot_res(weights,loss,accuracy)

	# different batch sizes
	train_batch_size()