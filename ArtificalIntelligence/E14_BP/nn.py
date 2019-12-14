import numpy as np

class FullyConnectedLayer(object):
    """
    Linear transformation: y = x W + b
    Input: (N, in_features), i.e. a row vector, in this example, N = 1
    Output: (N, out_features)

    Attributes:
      Weight: (in_features, out_features)
      Bias: (out_features)

    Ref:
    http://cs231n.stanford.edu/vecDerivs.pdf
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        """
        Xavier initialization
        # https://www.deeplearning.ai/ai-notes/initialization/
        W^{[l]} &\sim \mathcal{N}(\mu=0,\sigma^2 = \frac{1}{n^{[l-1]}})
        b^{[l]} &= 0
        """
        self.weight = np.random.normal(0,np.sqrt(1/in_features),(in_features,out_features))
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = None

    def forward(self, inputs):
        if type(self.bias) != type(None):
            return np.dot(inputs, self.weight) + self.bias
        else:
            return np.dot(inputs, self.weight)

    def __call__(self,x):
        return self.forward(x)

class Network(object):

    def __init__(self,in_features,hidden_features,out_features,learning_rate=0.001):
        self.fc1 = FullyConnectedLayer(in_features,hidden_features,True)
        self.fc2 = FullyConnectedLayer(hidden_features,out_features,True)
        self.learning_rate = learning_rate
        self.memory = {}
        self.train_flag = True

    def train(self):
        self.train_flag = True

    def eval(self):
        self.train_flag = False

    def relu(self,x):
        return np.maximum(0,x)

    def sigmoid(self,x):
        """
        Element-wise function
        \Sigma(x) = 1/(1+\ee^{-x})
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def MSE(self,y_hat,y):
        """
        Mean-square error (MSE)
        """
        return np.linalg.norm(y_hat - y) # 2-norm

    def forward(self,x):
        """
        w/o activation: z^{(l+1)} = W^{(l)}a^{(l)} + b^{(l)}
        w/ activation : a^{(l+1)} = f(z^{(l+1)})
        """
        if self.train_flag:
            self.memory["a0"] = np.copy(x)
            # print(x)
            x = self.fc1(x) # N * hidden
            # print(x)
            self.memory["z1"] = np.copy(x)
            x = self.sigmoid(x)
            # print(x,self.fc2.weight)
            self.memory["a1"] = np.copy(x)
            x = self.fc2(x) # N * out
            # print(x)
            self.memory["z2"] = np.copy(x)
            x = self.sigmoid(x)
        else: # test
            x = self.fc1(x) # N * hidden
            x = self.sigmoid(x)
            x = self.fc2(x) # N * out
            x = self.sigmoid(x)
        return x

    def backward(self,y_hat,y):
        """
        Use MSE as error function

        Ref: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
        """
        # Calculate \delta
        # output layer: \delta(n_l) = -(y - a(n_l)) * f'(z(n_l))
        # other layers: \delta(l) = W(l)^T\delta(l+1) * f'(z(l))
        delta = [0] * 3
        delta[2] = (y_hat - y) * self.d_sigmoid(self.memory["z2"]) # N * out_features
        delta[1] = np.dot(delta[2],self.fc2.weight.T) * self.d_sigmoid(self.memory["z1"]) # N * hidden_features

        # Calculat \nabla
        # output layer: \nabla_{W(l)}J(W,b;x,y) = \delta(l+1)(a(l))^T # outer product
        # other layers: \nabla_{b(l)}J(W,b;x,y) = \delta(l+1)
        nabla_W = [0] * 2
        nabla_W[1] = np.einsum("ij,ik->ikj",delta[2],self.memory["a1"]) # N * hidden_features * out_features
        nabla_W[0] = np.einsum("ij,ik->ikj",delta[1],self.memory["a0"]) # N * in_features * hidden_features
        nabla_b = [0] * 2
        nabla_b[1] = delta[2] # N * out_features
        nabla_b[0] = delta[1] # N * hidden_features

        # Update parameters
        # W(l) = W(l) - \alpha((1/m \Delta W(l)) + \lambda W(l))
        # b(l) = b(l) - \alpha(1/m \Delta b(l))
        # Use einsum to accelerate
        # https://rockt.github.io/2018/04/30/einsum
        nabla_W[1] = nabla_W[1].mean(axis=0)
        nabla_W[0] = nabla_W[0].mean(axis=0)
        nabla_b[1] = nabla_b[1].mean(axis=0)
        nabla_b[0] = nabla_b[0].mean(axis=0)

        self.fc2.weight = self.fc2.weight - self.learning_rate * nabla_W[1]
        self.fc1.weight = self.fc1.weight - self.learning_rate * nabla_W[0]
        self.fc2.bias = self.fc2.bias - self.learning_rate * nabla_b[1]
        self.fc1.bias = self.fc1.bias - self.learning_rate * nabla_b[0]