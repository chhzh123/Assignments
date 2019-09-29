import numpy as np

def normal_distribution(mu,sigma,size=10):
	"""
	Generate d-dimensional normal distribution N(mu,Sigma)
	mu: d-dim vector
	Sigma: d*d-dim covariance matrix
	n: number of generated points
	"""
	# return np.random.multivariate_normal(mu,sigma,size)
	return np.dot(np.random.randn(size,mu.size), np.linalg.cholesky(sigma)) + mu

def discriminant(x,mu,sigma,p_omega):
	"""
	g(x) = ln p(x|omega) + ln P(omega)
	"""
	d = mu.size
	return -1/2 * Manhalanobis(x,mu,sigma) - d/2 * np.log(np.pi) - 1/2 * np.log(np.abs(np.linalg.det(sigma))) + np.log(p_omega)

def L2(p1,p2):
	"""
	Euclidean distance (L2 distance)
	"""
	# return np.linalg.norm(p1-p2)
	return np.sqrt(np.sum(np.power(p1-p2,2)))

def Manhalanobis(x,mu,sigma):
	"""
	Given covariance matrix Sigma, compute the Manhalanobis distance
	from point x to mean mu
	"""
	return (x-mu).T.dot(np.linalg.inv(sigma)).dot((x-mu))

# v1 = np.random.rand(2)
# v2 = np.random.rand(2)
# mu = (v1+v2)/2
# m = np.stack((v1,v2),axis=0)
m = np.random.rand(4,10)
mu = np.mean(m,axis=1)
sig = np.cov(m)
v1 = m.T[1]
v2 = m.T[2]
print(normal_distribution(mu,sig))
print(discriminant(v1,mu,sig,1/2))
print(L2(v1,v2))
print(Manhalanobis(v1,mu,sig))