import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

MNIST_PATH = "./mnist/mnist.npz"

M_train = 60000
M_test = 10000
n = 28 * 28
C = 10
MAX_ITER = 5000

data = np.load(MNIST_PATH)
X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]
X_train.resize((M_train,n))
y_train.resize((M_train,1))
X_test.resize((M_test,n))
y_test.resize((M_test,1))

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

def normalize(Xin):
	Xin = Xin.astype(np.float64)
	Xin -= np.mean(Xin,axis=1,keepdims=True)
	Xin /= np.std(Xin,axis=1,keepdims=True)
	return Xin

X_train = normalize(X_train)
X_test = normalize(X_test)

class SoftmaxRegression():

	def train(self, X, y_true, n_classes, n_iters=10, learning_rate=0.1):
		self.n_samples, n_features = X.shape
		self.n_classes = n_classes
		
		self.weights = np.random.rand(self.n_classes, n_features)
		self.bias = np.zeros((1, self.n_classes))
		all_losses = []
		all_accuracy = []
		
		for i in range(n_iters):
			scores = self.compute_scores(X)
			probs = self.softmax(scores)
			y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
			y_one_hot = self.one_hot(y_true)

			loss = self.nll_loss(y_one_hot, probs)
			all_losses.append(loss)

			dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
			db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

			self.weights = self.weights - learning_rate * dw.T
			self.bias = self.bias - learning_rate * db

			if i % 100 == 0 or i == n_iters - 1:
				y_predict = self.predict(X_test)
				all_accuracy.append((np.sum(y_predict == y_test)/X_test.shape[0]) * 100)
				print(f'Iteration number: {i}, loss: {np.round(loss, 4)}, accuracy: {all_accuracy[-1]}%')

		return self.weights, self.bias, all_losses, all_accuracy

	def predict(self, X):
		scores = self.compute_scores(X)
		probs = self.softmax(scores)
		return np.argmax(probs, axis=1)[:, np.newaxis]

	def softmax(self, scores):
		exp = np.exp(scores) # (n_samples, n_classes)
		sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
		softmax = exp / sum_exp
		return softmax # (n_samples, n_classes)

	def compute_scores(self, X):
		"""
		    X: (n_samples, n_features)
			scores: (n_samples, n_classes)
		"""
		return np.dot(X, self.weights.T) + self.bias

	def nll_loss(self, y_true, probs):
		loss = - (1 / self.n_samples) * np.sum(y_true * np.log(probs))
		return loss

	def one_hot(self, y):
		one_hot = np.zeros((self.n_samples, self.n_classes))
		one_hot[np.arange(self.n_samples), y.T] = 1
		return one_hot

def plot_res(loss,accuracy):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	lns1 = ax.plot(loss, label="NLLLoss")
	ax.set_xlabel("Number of iterations (k)")
	ax.set_ylabel("Loss")
	ax2 = ax.twinx()
	x = np.arange(0,MAX_ITER+1,100)
	lns2 = ax2.plot(x,accuracy, "r", label="Accuracy")
	ax2.set_ylabel("Accuracy")
	yticks = mtick.FormatStrFormatter("%d%%")
	ax2.yaxis.set_major_formatter(yticks)
	
	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax.legend(lns,labs,loc=0)

	plt.show()


if __name__ == '__main__':

	time_start = time.time()
	regressor = SoftmaxRegression()
	w_trained, b_trained, loss, accuracy = regressor.train(X_train, y_train, learning_rate=0.1, n_iters=MAX_ITER, n_classes=C)
	time_end = time.time()
	print("Training time: {}s".format(time_end-time_start))

	plot_res(loss,accuracy)