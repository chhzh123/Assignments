# As usual, a bit of setup
# import numpy as np
# import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
# from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
# from cs231n.layers import *
# from cs231n.fast_layers import *
from cs231n.solver import Solver

# reading the input images
data = get_CIFAR10_data()
for k, v in data.items():
    print('%s: ' % k, v.shape)

# set up model
model = ThreeLayerConvNet(hidden_dim=512, reg=0.001)

# encapsulate it into solver
solver = Solver(model, data,
                num_epochs=3,
                batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True,
                print_every=50,
                checkpoint_name="checkpoint/cnn")
solver.train()