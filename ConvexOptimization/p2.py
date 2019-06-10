# Optimization: Logistic regression (MNIST)
# Hongzheng Chen 17341015

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

MNIST_PATH = "mnist.npz"

# constants
n = 28 * 28
M = 60000 # training set
MTest = 10000 # test set
max_iter = 10000
m = 100 # mini-batch size
C = 10
alpha = 10e-2
acc = 10e-7
# X: M*n
# Y: M
# W: C*n

res = []
res_acc = []

##### gradient descent #####

def softmax(W,x,y):
	sume = 0
	dot = []
	for c in range(C):
		dot.append(np.dot(W[c],x))
	max_dot = max(dot) # avoid overflow
	for c in range(C):
		sume += np.exp(dot[c]-max_dot)
	p = np.exp(dot[y]-max_dot) / sume
	return p

def grad(W,X,Y,j):
	dw = np.zeros(n)
	minibatch = random.sample(range(M),m)
	for i in minibatch:
		p = softmax(W,X[i],Y[i])
		dw -= X[i] * ((Y[i] == j) - p)
	return dw

def gradient_descent(Wk):
	t = 0
	alphak = alpha
	while True:
		# alphak = alpha
		Wk_new = np.zeros((C,n))
		for j in range(C):
			Wk_new[j] = Wk[j] - alphak / m * grad(Wk,X,Y,j)
		diff = np.linalg.norm(Wk_new - Wk, ord=2)
		if diff < acc or t > max_iter:
			break
		Wk = Wk_new.copy()
		t += 1
		if t % 100 == 0:
			print("Iteration: {} \tDiff: {}".format(t,diff))
			accuracy = test(Wk)
			res.append(Wk)
			res_acc.append(accuracy)
			alphak = alphak * 0.99
	print(t)
	return Wk

##### inference #####

def inference(W,x):
	max_index = 0
	max_p = 0
	for c in range(C):
		p = softmax(W,x,c)
		if p > max_p:
			max_index = c
			max_p = p
	return max_index

def test(W):
	print("Begin testing...")
	accuracy = 0
	for i in range(MTest):
		y_pred = inference(W,XTest[i])
		if (YTest[i] == y_pred):
			accuracy += 1
	accuracy /= MTest
	print("Accuracy: {}%".format(accuracy*100))
	return accuracy

##### plot results #####

def plot_res(res,res_acc):
	res = np.array(res)
	res_acc = np.array(res_acc) * 100
	opt_dist = np.zeros(len(res))
	for i in range(len(res)):
		opt_dist[i] = np.linalg.norm(res[i] - res[-1], ord=2)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	lns1 = ax.plot(opt_dist, label='$||x^{(k)}-x^\\star|||_2^2$')
	ax.set_xlabel("k")
	ax.set_ylabel("Distance")
	ax2 = ax.twinx()
	lns2 = ax2.plot(res_acc, "r", label="Accuracy")
	ax2.set_ylabel("Accuracy")
	yticks = mtick.FormatStrFormatter("%d%%")
	ax2.yaxis.set_major_formatter(yticks)

	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax.legend(lns,labs,loc=0)

	plt.show()

def normalize(Xin):
	Xin = Xin.astype(np.float64)
	Xin -= np.mean(Xin,axis=1,keepdims=True)
	Xin /= np.std(Xin,axis=1,keepdims=True)
	return Xin

if __name__ == '__main__':
	data = np.load(MNIST_PATH)
	X, Y = data["x_train"], data["y_train"]
	XTest, YTest = data["x_test"], data["y_test"]
	data.close()

	X.resize((M,n))
	XTest.resize((MTest,n))
	X = normalize(X)
	XTest = normalize(XTest)
	Wk = np.zeros((C,n))
	Wk = np.random.normal(0,1,(C,n))
	print("Begin training...")
	gradient_descent(Wk)

	plot_res(res,res_acc)