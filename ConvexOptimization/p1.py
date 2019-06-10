# Optimization: LASSO
# Hongzheng Chen 17341015

import numpy as np
import matplotlib.pyplot as plt

# constants
m = 50
n = 100
sd = 5 # sparse degree
acc = 10e-9
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

res = []

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

def proxgrad(x0):
	t = 0
	xk = x0.copy()
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

def admm(x0,y0,v0):
	t = 0
	xk = x0.copy()
	yk = y0.copy()
	vk = v0.copy()
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

def subgrad(x0):
	xk = x0.copy()
	t = 0
	while True:
		pdx = np.zeros(xk.size)
		alphak = alpha / (t + 1) # remember to decay the step
		for i in range(xk.size):
			if xk[i] != 0:
				pdx[i] = 1 if xk[i] > 0 else -1
			else:
				pdx[i] = 2 * np.random.random() - 1 # pick a random float from [-1,1]
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

##### main function #####

if __name__ == '__main__':
	print(x)

	# Proximal Gradient
	x0 = np.zeros(n)
	print(proxgrad(x0))
	plot_res(res,"Proximal Gradient")

	# ADMM
	res = []
	x0 = np.zeros(n)
	y0 = np.zeros(n)
	v0 = np.zeros(n)
	print(admm(x0,y0,v0))
	plot_res(res,"ADMM")

	# Subgradient
	res = []
	x0 = np.zeros(n)
	print(subgrad(x0))
	plot_res(res,"Subgradient")