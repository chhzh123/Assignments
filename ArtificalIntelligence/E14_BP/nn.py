import numpy as np

class FullyConnectedLayer(object):
    """
    Linear transformation: y = x W^T + b
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

        Kaiming He initialization
        # https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899
        """
        self.weight = np.random.normal(0,np.sqrt(2/in_features),(out_features,in_features))
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = None

    def forward(self, inputs):
        """
        Forward propagation
        """
        if type(self.bias) != type(None):
            return np.dot(inputs, self.weight.T) + self.bias
        else:
            return np.dot(inputs, self.weight.T)

    def __call__(self,x):
        """
        Syntax sugar for forward method
        """
        return self.forward(x)

class Network(object):

    def __init__(self,in_features,hidden_features,out_features,learning_rate=0.01):
        """
        Here three-layer network architecture is used

        The number of neurons in each layer is listed below:
        in_features -> hidden_features -> out_features
        """
        self.fc1 = FullyConnectedLayer(in_features,hidden_features,True)
        self.fc2 = FullyConnectedLayer(hidden_features,out_features,True)
        self.learning_rate = learning_rate
        self.memory = {} # used for store intermediate results
        self.train_flag = True

    def train(self):
        """
        When training, memory is set to remember the intermediate results
        """
        self.train_flag = True

    def eval(self):
        """
        When inferencing, memory is no need to set
        """
        self.train_flag = False

    def relu(self,x):
        """
        Relu(x) = x, x > 0
                  0, x <= 0
        """
        return np.maximum(0,x)

    def d_relu(self,x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self,x):
        """
        Element-wise function
        \Sigma(x) = 1/(1+\ee^{-x})
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self,x):
        """
        Derivative of sigmoid function
        \Sigma'(x) = \Sigma(x) * (1 - \Sigma(x))
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self,x):
        return np.tanh(x)

    def d_tanh(self,x):
        return 1 - np.tanh(x) ** 2

    def MSE(self,y_hat,y):
        """
        Mean-square error (MSE)
        """
        return np.linalg.norm(y_hat - y) # 2-norm

    def cross_entropy(self,y_hat,y):
        """
        Cross entropy loss
        """
        return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

    def forward(self,x):
        """
        w/o activation: z^{(l+1)} = W^{(l)}a^{(l)} + b^{(l)}
        w/ activation : a^{(l+1)} = f(z^{(l+1)})
        """
        # training
        if self.train_flag:
            self.memory["a0"] = np.copy(x)
            x = self.fc1(x) # N * hidden
            self.memory["z1"] = np.copy(x)
            x = self.sigmoid(x)
            self.memory["a1"] = np.copy(x)
            x = self.fc2(x) # N * out
            self.memory["z2"] = np.copy(x)
            x = self.sigmoid(x)
        # inferencing
        else:
            x = self.fc1(x) # N * hidden
            x = self.sigmoid(x)
            x = self.fc2(x) # N * out
            x = self.sigmoid(x)
        return x

    def backward(self,y_hat,y,lamb=0):
        """
        Use Mean-Squared Error (MSE) as error function

        lambda is used for weight decay

        Ref: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
        """
        batch_size = y.shape[0]
        # Calculate \delta
        # output layer: \delta(n_l) = -(y - a(n_l)) * f'(z(n_l))
        # other layers: \delta(l) = W(l)^T\delta(l+1) * f'(z(l))
        delta = [0] * 3
        delta[2] = (y_hat - y) * self.d_sigmoid(self.memory["z2"]) # N * out_features
        delta[1] = np.dot(delta[2],self.fc2.weight) * self.d_sigmoid(self.memory["z1"]) # N * hidden_features
        # print(delta[2].shape,delta[1].shape)

        # Calculat \nabla
        # output layer: \nabla_{W(l)}J(W,b;x,y) = \delta(l+1)(a(l))^T # outer product
        # other layers: \nabla_{b(l)}J(W,b;x,y) = \delta(l+1)
        nabla_W = [0] * 2
        nabla_W[1] = np.einsum("ij,ik->ijk",delta[2],self.memory["a1"]) # N * out_features * hidden_features
        nabla_W[0] = np.einsum("ij,ik->ijk",delta[1],self.memory["a0"]) # N * hidden_features * in_features
        nabla_b = [0] * 2
        nabla_b[1] = delta[2] # N * out_features
        nabla_b[0] = delta[1] # N * hidden_features
        # print(nabla_W[1].shape,nabla_W[0].shape,nabla_b[1].shape,nabla_b[0].shape)

        # Update parameters
        # W(l) = W(l) - \alpha((1/m \Delta W(l)) + \lambda W(l))
        # b(l) = b(l) - \alpha(1/m \Delta b(l))
        # Use einsum to accelerate
        # https://rockt.github.io/2018/04/30/einsum
        nabla_W[1] = nabla_W[1].mean(axis=0)
        nabla_W[0] = nabla_W[0].mean(axis=0)
        nabla_b[1] = nabla_b[1].mean(axis=0)
        nabla_b[0] = nabla_b[0].mean(axis=0)

        # weight decay, lambda is the L2 regularization term
        self.fc2.weight -= self.learning_rate * (nabla_W[1] + lamb * self.fc2.weight / batch_size)
        self.fc1.weight -= self.learning_rate * (nabla_W[0] + lamb * self.fc1.weight / batch_size)
        self.fc2.bias -= self.learning_rate * nabla_b[1]
        self.fc1.bias -= self.learning_rate * nabla_b[0]