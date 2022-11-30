import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class PINNSolver():
    def __init__(self, model, residual_sample_points):
        self.model = model

        self.t = residual_sample_points[:, 0:1]
        self.x = residual_sample_points[:, 1:2]

        self.history = []
        self.iterations = 0

    def get_residual(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t)
            tape.watch(self.x)

            u = self.model(tf.stack([self.t[:, 0], self.x[:, 0]], axis=1))
            u_x = tape.gradient(u, self.x)

        u_t = tape.gradient(u, self.t)
        u_xx = tape.gradient(u_x, self.x)

        del tape

        return self.get_raw_residual(self.t, self.x, u, u_t, u_x, u_xx)

    def loss(self, X, u):
        r = self.get_residual()
        phi_r = tf.reduce_mean(tf.square(r))

        loss = phi_r

        for i in range(len(X)):
            u_pred = self.model(X[i])
            loss += tf.reduce_mean(tf.square(u[i] - u_pred))

        return loss

    def get_gradient(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss(X, u)

        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, g

    def get_raw_residual(self, t, x, u, u_t, u_x, u_xx):
        viscosity = 5
        return u_t + u * u_x - viscosity * u_xx

    def solve(self, optimizer, X, u, N=1001):
        @tf.function
        def train_step():
            loss, grad_theta = self.get_gradient(X, u)

            optimizer.apply_gradients(
                zip(grad_theta, self.model.trainable_variables))
            return loss

        for _ in range(N):
            loss = train_step()

            self.current_loss = loss.numpy()
            self.callback()

    def callback(self):
        if self.iterations % 50 == 0:
            print('Iteration {:05d}: loss = {:10.8e}'.format(
                self.iterations, self.current_loss))
        self.history.append(self.current_loss)
        self.iterations += 1

    def plot_solution(self, **kwargs):
        N = 600
        t = np.linspace(
            self.model.lower_bounds[0], self.model.upper_bounds[0], N+1)
        x = np.linspace(
            self.model.lower_bounds[1], self.model.upper_bounds[1], N+1)
        T, X = np.meshgrid(t, x)
        x_grid = np.vstack([T.flatten(), X.flatten()]).T
        u_prediction = self.model(tf.cast(x_grid, "float32"))
        U = u_prediction.numpy().reshape(N+1, N+1)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, X, U, cmap='viridis', **kwargs)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        ax.view_init(35, 35)
        plt.show()

    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.history)), self.history, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        plt.show()
