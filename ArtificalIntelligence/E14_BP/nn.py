import numpy as np

class FullyConnectedLayer(object):
    """
    Linear transformation: y = x W + b
    Input: (N, in_features), i.e. a row vector, in this example, N = 1
    Output: (N, out_features)

    Attributes:
      Weight: (in_features, out_features)
      Bias: (out_features)

    By matrix calculus, we have
    \pd{y}{x} = W

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
        # self.weight = np.random.rand(in_features,out_features)
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
        self.fc1 = FullyConnectedLayer(in_features,hidden_features,False)
        self.fc2 = FullyConnectedLayer(hidden_features,out_features,False)
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
        \Sigma(x) = 1/(1+\ee^{-x})
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self,x):
        """
        f(x_i) = e^{x_i} / \sum_j e^{x_j}
        """
        return np.exp(x) / np.sum(np.exp(x),axis=0)

    def MSE(self,y_hat,y):
        return np.linalg.norm(y_hat - y)

    def forward(self,x):
        if self. train_flag:
            self.memory["x0"] = np.copy(x)
            # print(x)
            x = self.fc1(x) # 1 * hidden
            # print(x)
            self.memory["Z1"] = np.copy(x)
            x = self.sigmoid(x)
            # print(x,self.fc2.weight)
            self.memory["x1"] = np.copy(x)
            x = self.fc2(x) # 1 * out
            # print(x)
            self.memory["Z2"] = np.copy(x)
            x = self.sigmoid(x)
            # print(x)
            self.memory["x2"] = np.copy(x)
        else: # test
            x = self.fc1(x) # 1 * hidden
            x = self.sigmoid(x)
            x = self.fc2(x) # 1 * out
            x = self.sigmoid(x)
        return x

    def backward(self,y_hat,y):
        """
        Use MSE as error function

        Ref: https://sudeepraja.github.io/Neural/
        """
        W2 = self.fc2.weight.copy()
        delta_2 = (y_hat - y) * self.d_sigmoid(self.memory["Z2"]) # 1 * out_features
        dW2 = np.dot(self.memory["x1"].reshape(-1,1),delta_2.reshape(1,-1)) # vector outer product -> hidden * out_features
        self.fc2.weight = self.fc2.weight - self.learning_rate * dW2
        delta_1 = np.dot(delta_2,W2.T) * self.d_sigmoid(self.memory["Z1"]) # 1 * hidden_features
        dW1 = np.dot(self.memory["x0"].reshape(-1,1),delta_1.reshape(1,-1)) # in_features * hidden
        self.fc1.weight = self.fc1.weight - self.learning_rate * dW1