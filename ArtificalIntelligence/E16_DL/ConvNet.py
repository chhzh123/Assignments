from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

data = get_CIFAR10_data()
for k, v in data.items():
    print('{}: '.format(k), v.shape)


# set up model
model = ThreeLayerConvNet(hidden_dim=512, reg=0.001)

# encapsulate it into solver
solver = Solver(model, data,
                num_epochs=1,
                batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True,
                print_every=50,
                checkpoint_name="checkpoint/cnn")
solver.train()


solver.check_accuracy(data["X_test"],data["y_test"])


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.plot(solver.loss_history)
fig.savefig("fig/loss.pdf",format="pdf")