import cv2
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

ti.cfg.arch = ti.cuda  # Run on GPU by default
output_every = 2

# some constants
n = 256 # grid size
steps = 256
max_steps = 512

c = 340 # speed of sound
alpha = 0.00000
dx = 1 / n
inv_dx = 1 / dx
dt = (np.sqrt(alpha ** 2 + dx ** 2 / 3) - alpha) / c
learning_rate = 1

# firstly declare the variables
u = ti.var(dt=ti.f32)
u_hat = ti.var(dt=ti.f32)
res = ti.var(dt=ti.f32)
loss = ti.var(dt=ti.f32)

@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.ij, n).place(u)
    ti.root.dense(ti.l, max_steps).dense(ti.ij, n).place(u.grad)
    ti.root.dense(ti.ij, n).place(u_hat)
    ti.root.dense(ti.ij, n).place(u_hat.grad)
    ti.root.dense(ti.ij, n).place(res)
    ti.root.dense(ti.ij, n).place(res.grad)
    ti.root.place(loss)
    ti.root.place(loss.grad)

@ti.kernel
def initialize():
    for i in range(n):
        for j in range(n):
            u[0, i, j] = res[i, j]

@ti.func
def laplace(t, i, j):
    return ti.sqr(inv_dx) * (-4 * u[t, i, j] + u[t, i, j - 1] + u[t, i, j + 1] + u[t, i + 1, j] + u[t, i - 1, j])


@ti.kernel
def fdtd(t: ti.i32):
    """
    The update formula comes from the paper DiffTaichi
    """
    for i in range(n):
        for j in range(n):
            u[t, i, j] = 2 * u[t - 1, i, j] \
                       + (ti.sqr(c) * ti.sqr(dt) + c * alpha * dt) * laplace(t - 1, i, j) \
                       - u[t - 2, i, j] \
                       - c * alpha * dt * laplace(t - 2, i, j)

@ti.kernel
def cal_loss(t: ti.i32):
    for i in range(n):
        for j in range(n):
            # make sure the atomicity when updating the same value in parallel
            ti.atomic_add(loss, ti.sqr(dx) * ti.sqr(u_hat[i, j] - u[t, i, j]))

@ti.kernel
def apply_grad():
    # gradient descent
    for i, j in res.grad:
        res[i, j] -= learning_rate * res.grad[i, j]

def forward():
    initialize()
    for t in range(2, steps):
        fdtd(t)
        # the following output framework references from Taichi examples
        if (t + 1) % output_every == 0:
            img = np.zeros(shape=(n, n), dtype=np.float32)
            for i in range(n):
                for j in range(n):
                    img[i, j] = u[t, i, j] + 0.5
            img = cv2.resize(img, fx=4, fy=4, dsize=None)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            img = np.clip(img, 0, 255)
            cv2.imwrite("output/{:04d}.png".format(t), img * 255)
    cal_loss(steps - 1)

def main():
    """
    Differentiable programming framework can be found in
    https://taichi.readthedocs.io/en/latest/syntax.html#kernels
    """
    # read in figures
    img = cv2.imread('erythrocyte.png')[:,:,0]
    # normalization
    img = img / 255.0
    img -= img.mean()
    img = cv2.resize(img, (n, n))
    for i in range(n):
        for j in range(n):
            u_hat[i, j] = float(img[i, j])

    losses = []
    for it in range(100):
        # encapsulate loss in taichi
        with ti.Tape(loss):
            forward()
        print('Iter {} Loss = {}'.format(it,loss[None]))
        losses.append(loss[None])
        # update gradient
        apply_grad()

    # output loss curve
    plt.set_xlabel("Iteration")
    plt.set_ylabel("Loss")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()