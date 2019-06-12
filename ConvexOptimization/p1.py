# Optimization: LASSO
# Hongzheng Chen 17341015

import numpy as np
import matplotlib.pyplot as plt

# constants
m = 50
n = 100
sd = 5 # sparse degree
accuracy = 1e-8
p = 1e-1

# initialization
A = np.random.normal(0,1,(m,n))
x = np.random.normal(0,1,n)
e = np.random.normal(0,0.1,m)
xIndex = np.random.randint(0,n,sd)
for i in range(n):
	x[i] = x[i] if i in xIndex else 0
b = np.dot(A, x) + e

class ProximalGradient():

	def __init__(self, alpha=1e-3):
		self.name = "Proximal Gradient"
		self.alpha = alpha

	def soft_thresholding(self, x, offset):
		if x < (-1) * offset:
			return x + offset
		elif x > offset:
			return x - offset
		else:
			return 0

	def prox(self, xk_old, offset):
		# v_soft_thresholding = np.vectorize(self.soft_thresholding)
		# return v_soft_thresholding(xk_old,offset)
		xk_new = np.zeros(xk_old.size)
		for i in range(xk_old.size):
			xk_new[i] = self.soft_thresholding(xk_old[i],offset)
		return xk_new

	def train(self,A,b,p):
		_, self.n = A.shape
		self.xk = np.zeros(self.n)

		res = []
		t = 0
		while True:
			xhat = self.xk - self.alpha * np.dot(A.T, np.dot(A, self.xk) - b)
			xk_new = self.prox(xhat, self.alpha * p)
			if np.linalg.norm(xk_new - self.xk, ord=2) < accuracy:
				break
			res.append(xk_new)
			self.xk = xk_new.copy()
			t += 1

		print(t)
		return self.xk, res

class ADMM():

	def __init__(self, c=1e-3):
		self.name = "ADMM"
		self.c = c

	def soft_thresholding(self, x, offset):
		if x < (-1) * offset:
			return x + offset
		elif x > offset:
			return x - offset
		else:
			return 0

	def prox(self, xk_old, offset):
		xk_new = np.zeros(xk_old.size)
		for i in range(xk_old.size):
			xk_new[i] = self.soft_thresholding(xk_old[i],offset)
		return xk_new

	def train(self,A,b,p):
		_, self.n = A.shape
		self.xk = np.zeros(self.n)
		self.yk = np.zeros(self.n)
		self.vk = np.zeros(self.n)

		res = []
		t = 0
		while True:
			xk_new = np.dot(
				np.linalg.inv(np.dot(A.T, A) + self.c * np.eye(self.n,self.n)),
				np.dot(A.T, b) + self.c * self.yk - self.vk)
			self.yk = self.prox(xk_new + self.vk / self.c, p / self.c)
			self.vk = self.vk + self.c * (xk_new - self.yk)
			if np.linalg.norm(xk_new - self.xk, ord=2) < accuracy:
				break
			res.append(xk_new)
			self.xk = xk_new.copy()
			t += 1

		print(t)
		return self.xk, res

class Subgradient():

	def __init__(self,alpha=1e-3):
		self.name = "Subgradient"
		self.alpha = alpha

	def subgrad(self,x):
		# subgradient of |x|
		pdx = np.zeros(x.size)
		for i in range(x.size):
			if x[i] != 0:
				pdx[i] = 1 if x[i] > 0 else -1
			else: # pick a random float from [-1,1]
				pdx[i] = 2 * np.random.random() - 1
		return pdx

	def train(self,A,b,p):
		_, self.n = A.shape
		self.xk = np.zeros(self.n)

		res = []
		t = 0
		while True:
			alphak = self.alpha / (t + 1) # remember to decay the step
			pdf = np.dot(A.T, np.dot(A, self.xk) - b) + self.subgrad(self.xk)
			xk_new = self.xk - alphak * pdf
			if np.linalg.norm(xk_new - self.xk, ord=2) < accuracy:
				break
			res.append(xk_new)
			self.xk = xk_new.copy()
			t += 1

		print(t)
		return self.xk, res

##### plot results #####

def plot_res(res,string=""):
	plt.title("Distance to optimal and true value ({})".format(string))
	res = np.array(res)
	opt_dist = np.zeros(len(res))
	true_dist = np.zeros(len(res))
	for i in range(len(res)):
		opt_dist[i] = np.linalg.norm(res[i] - res[-1], ord=2)
		true_dist[i] = np.linalg.norm(res[i] - x, ord=2)
	plt.plot(opt_dist, label='$||x^{(k)}-x^\\star||_2$')
	plt.plot(true_dist, label='$||x^{(k)}-x_{true}||_2$')
	plt.xlabel("Number of iterations (k)")
	plt.ylabel("Distance")
	plt.legend(loc=1)
	plt.show()

def plot_reg(ax,res,p,string=""):
	res = np.array(res)
	opt_dist = np.zeros(len(res))
	true_dist = np.zeros(len(res))
	for i in range(len(res)):
		opt_dist[i] = np.linalg.norm(res[i] - res[-1], ord=2)
		true_dist[i] = np.linalg.norm(res[i] - x, ord=2)
	ax[0].plot(opt_dist, label="p={:.4}".format(p))
	ax[1].plot(true_dist, label="p={:.4}".format(p))

##### test functions #####

def test(fun,A,b,p):
	method = fun()
	xk, res = method.train(A,b,p)
	plot_res(res,method.name)

def test_reg(fun,A,b):
	method = fun()
	fig, ax = plt.subplots(2,1)
	ax[0].set_title("Effect of regularization parameter ({})".format(method.name))
	ax[0].set_xlabel("Number of iterations (k)")
	ax[0].set_ylabel("$||x^{(k)}-x^\\star||_2$")
	ax[1].set_xlabel("Number of iterations (k)")
	ax[1].set_ylabel("$||x^{(k)}-x_{true}||_2$")

	p = 2
	for i in range(4):
		p *= 0.5
		xk, res = method.train(A,b,p)
		plot_reg(ax,res,p)

	ax[0].legend(loc=1)
	ax[1].legend(loc=1)
	plt.tight_layout()
	plt.show()

##### main function #####

if __name__ == '__main__':

	test(ProximalGradient,A,b,p)
	test(ADMM,A,b,p)
	test(Subgradient,A,b,p)

	accuracy = 1e-6
	test_reg(ProximalGradient,A,b)
	test_reg(ADMM,A,b)
	test_reg(Subgradient,A,b)