
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys

dataset = pd.read_csv("football.txt",sep=",")
dataset.head()


# In[ ]:


def prob(x, mu, sigma):
    """
    Calculate the Gaussian distribution

    N(x|\mu,\Sigma)=\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\{-\frac{1}{2}(x-\mu)^T}\Sigma^{-1}(x-\mu)\}

    Notice the input x here should be a 1-d vector
    """
    if x.ndim != 1 or mu.ndim != 1:
        raise RuntimeError("Dimension error!")
    # print(x,mu,sigma)
    D = x.shape[0]
    expOn = - 1 / 2 * np.matmul(np.matmul((x - mu).T,np.linalg.inv(sigma)),x - mu)
    divBy = np.power(2 * np.pi, D / 2) * np.sqrt(np.linalg.det(sigma))
    return np.exp(expOn) / divBy

def EM(dataMat, n_components=3, maxIter=100):
    """
    Expectation-Maximization (EM) algorithm
    
    This implementation has been extended to support different number of components
    """
    n_samples, D = np.shape(dataMat) # n_samples=16 D=7
    
    # 1. Initialize Gaussian parameters
    pi_k = np.ones(n_components) / n_components # mixing coefficients
    # randomly select k(n_components) samples as the mean of each class
    # choices = np.random.choice(n_samples,n_components)
    # predefined mean of each class
    choices = [1,13,11]
    print([(i,dataset["Country"][i]) for i in choices])
    mu_k = np.array([dataMat[i,:] for i in choices]).reshape(n_components,D) # k * D
    # mu_k = [dataMat[5, :], dataMat[21, :], dataMat[26, :]]
    sigma_k = [np.eye(D) for x in range(n_components)] # k * D * D

    gamma_k = np.zeros((n_samples, n_components)) # n * k
    # Iterate for maxIter times
    for i in range(maxIter):
        """
        2. E step
        \gamma(z_{nk}) = \frac{\pi_k prob(x_n|\mu_k,\Sigma_k)}{sum_pi_mul}
        sum_pi_mul = \sum_{j=1}^K \pi_j prob(x_n|\mu_j,\Sigma_j)
        """
        for n in range(n_samples):
            sum_pi_mul = 0 # denominator
            for k in range(n_components):
                gamma_k[n, k] = pi_k[k] * prob(dataMat[n, :], mu_k[k], sigma_k[k])
                sum_pi_mul += gamma_k[n, k]
            # normalization
            for k in range(n_components):
                gamma_k[n, k] /= sum_pi_mul
        # summarize gamma_k along different samples
        N_k = np.sum(gamma_k, axis=0)

        """
        3. M step
        \mu_k^{new} = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n
	    \Sigma_k^{new} = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(x_n-\mu_k^{new})(x_n-\mu_k^{new})^T
	    \pi_k^{new} = \frac{N_k}{N}

        N_k=\sum_{n=1}^N\gamma(z_{nk})
        """
        for k in range(n_components):
            # Calculate \mu_k
            mu_k[k] = np.zeros(D,dtype=np.float64)
            for n in range(n_samples):
                mu_k[k] += gamma_k[n, k] * dataMat[n, :]
            mu_k[k] /= N_k[k]

            # Calculate \Sigma_k
            sigma_k[k] = np.zeros((D, D),dtype=np.float64)
            for n in range(n_samples):
                sigma_k[k] += gamma_k[n, k] * np.matmul((dataMat[n, :] - mu_k[k]).reshape(1,-1).T, (dataMat[n, :] - mu_k[k]).reshape(1,-1)) # be careful of outer product!
            sigma_k[k] /= N_k[k]
            
            # Calculate new mixing coefficient
            pi_k[k] = N_k[k] / n_samples

        sigma_k += np.eye(D)

    print("gamma: ",gamma_k)
    print("mu: ",mu_k)
    print("Sigma: ",sigma_k)

    return gamma_k


# In[ ]:


def gaussianCluster(dataMat, n_components, max_iter):
    n_samples, D = np.shape(dataMat)
    centroids = np.zeros((n_components, D))
    gamma = EM(dataMat,n_components,max_iter)

    # get the cluster result
    clusterAssign = np.zeros((n_samples, 2))
    for n in range(n_samples):
        clusterAssign[n, :] = np.argmax(gamma[n, :]), np.amax(gamma[n, :])

    # calculate the final results
    for k in range(n_components):
        pointsInCluster = dataMat[np.nonzero(clusterAssign[:, 0] == k)[0]]
        centroids[k, :] = np.mean(pointsInCluster, axis=0)
    return centroids, clusterAssign[:,0]


# In[ ]:


from sklearn import mixture

def sklearn_em(data,n_components,max_iter):
    clst = mixture.GaussianMixture(n_components=n_components,max_iter=max_iter,covariance_type="full")
    clst.fit(data)
    predicted_labels = clst.predict(data)
    return clst.means_, predicted_labels


# In[ ]:


import matplotlib.pyplot as plt

def showCluster(dataset, k, centroids, clusterAssment):
    numSamples, dim = dataset.shape

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i])
        plt.plot(dataset[i, 0], dataset[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


# In[ ]:


n_components = 3
data = dataset[dataset.columns.values[dataset.columns.values != "Country"]].to_numpy()
# centroids, labels = sklearn_em(data,n_components,100)
centroids, labels = gaussianCluster(data.astype(np.float64),n_components,100)
showCluster(data, n_components, centroids, labels)
res = {0:[],1:[],2:[]}
for i,label in enumerate(labels):
    res[label].append(dataset["Country"][i])
for key in res:
    print(key,*res[key])

