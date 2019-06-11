# Optimization: LASSO
# Hongzheng Chen 17341015

import numpy as np
import matplotlib.pyplot as plt

# constants
m = 50
n = 100
sd = 5 # sparse degree
acc = 10e-8
p = 10e-3
alpha = 10e-4 # prox
c = 10e-4 # admm

# initialization
A = np.random.normal(0,1,(m,n))
x = np.random.normal(0,1,n)
e = np.random.normal(0,0.1,m)
xIndex = np.random.randint(0,n,sd)
for i in range(n):
	x[i] = x[i] if i in xIndex else 0
b = np.dot(A, x) + e

###### proximal gradient descent #####

def soft_thresholding(x,offset):
	if x < (-1)*offset:
		return x + offset
	elif x > offset:
		return x - offset
	else:
		return 0

def prox(xk_old,offset):
	xk_new = np.zeros(xk_old.size)
	for i in range(xk_old.size):
		xk_new[i] = soft_thresholding(xk_old[i],offset)
	return xk_new

def proxgrad(res):
	t = 0
	xk = np.zeros(n)
	while True:
		xhat = xk - alpha * np.dot(A.T, np.dot(A, xk) - b)
		xk_new = prox(xhat, alpha * p)
		if np.linalg.norm(xk_new - xk, ord=2) < acc:
			break
		res.append(xk_new)
		xk = xk_new.copy()
		t += 1
	print(t)
	return xk

##### alternating direction method of multipliers #####

def admm(res):
	t = 0
	xk = np.zeros(n)
	yk = np.zeros(n)
	vk = np.zeros(n)
	while True:
		xk_new = np.dot(
			np.linalg.inv(np.dot(A.T, A) + c * np.eye(n,n)),
			np.dot(A.T, b) + c * yk - vk)
		yk_new = prox(xk_new + vk / c, p / c)
		vk_new = vk + c * (xk_new - yk_new)
		if np.linalg.norm(xk_new - xk, ord=2) < acc:
			break
		res.append(xk_new)
		xk = xk_new.copy()
		yk = yk_new.copy()
		vk = vk_new.copy()
		t += 1
	print(t)
	return xk

##### subgradient #####

def subgrad(res):
	xk = np.zeros(n)
	t = 0
	while True:
		pdx = np.zeros(xk.size)
		alphak = alpha / (t + 1) # remember to decay the step
		for i in range(xk.size):
			if xk[i] != 0:
				pdx[i] = 1 if xk[i] > 0 else -1
			else: # pick a random float from [-1,1]
				pdx[i] = 2 * np.random.random() - 1
		pdf = np.dot(A.T, np.dot(A,xk) - b) + pdx
		xk_new = xk - alphak * pdf
		if np.linalg.norm(xk_new - xk, ord=2) < acc:
			break
		res.append(xk_new)
		xk = xk_new.copy()
		t += 1
	print(t)
	return xk

##### plot results #####

def plot_res(res,string=""):
	plt.title("Distance to optimal and true value ({})".format(string))
	res = np.array(res)
	opt_dist = np.zeros(len(res))
	true_dist = np.zeros(len(res))
	for i in range(len(res)):
		opt_dist[i] = np.linalg.norm(res[i] - res[-1], ord=2)
		true_dist[i] = np.linalg.norm(res[i] - x, ord=2)
	plt.plot(opt_dist, label='$||x^{(k)}-x^\\star|||_2^2$')
	plt.plot(true_dist, label='$||x^{(k)}-x_{true}||_2^2$')
	plt.xlabel("k")
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

def test(fun,string=""):
	res = []
	print(proxgrad(res))
	plot_res(res,string)

def test_reg(fun,string=""):
	global p
	fig, ax = plt.subplots(2,1)
	ax[0].set_title("Effect of regularization parameter ({})".format(string))
	ax[0].set_xlabel("k")
	ax[0].set_ylabel("$||x^{(k)}-x^\\star|||_2^2$")
	ax[1].set_xlabel("k")
	ax[1].set_ylabel("$||x^{(k)}-x_{true}||_2^2$")

	p = 2
	for i in range(4):
		p = p * 0.5
		res = []
		fun(res)
		plot_reg(ax,res,p)

	ax[0].legend(loc=1)
	ax[1].legend(loc=1)
	plt.tight_layout()
	plt.show()

##### main function #####

if __name__ == '__main__':

	# test(proxgrad,"Proximal Gradient")
	# test(admm,"ADMM")
	# test(subgrad,"Subgradient")

	test_reg(proxgrad,"Proximal Gradient")
	test_reg(admm,"ADMM")
	test_reg(subgrad,"Subgradient")