# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
from pinn import PINN
from pinn_solver import PINNSolver
import time

tf.keras.backend.set_floatx("float32")

# Define initial condition


def fun_u_0(x):
    pi = tf.constant(np.pi, dtype="float32")
    return tf.sin(pi * x)


# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n, 1), dtype="float32")

# Define residual of the PDE
# ohhhh --- this is the residual because burgers equation is equal to zero
# anything greater than this is not physically possible and is an error


def fun_r(t, x, u, u_t, u_x, u_xx):
    # all these parameters are gradients of U
    # we approximate U with a neural network
    viscosity = 5
    return u_t + u * u_x - viscosity * u_xx


# Set number of data points
N_0 = 50
N_b = 50
N_r = 10000

# Set boundary
tmin = 0.
tmax = 1.
xmin = 0.
xmax = 1.

# Lower bounds
# creates a constant -- variable
lb = tf.constant([tmin, xmin], dtype="float32")
# Upper bounds
ub = tf.constant([tmax, xmax], dtype="float32")

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw uniform sample points for initial boundary data
# initial condition for the time
t_0 = tf.ones((N_0, 1), dtype="float32")*lb[0]  # time
x_0 = tf.random.uniform(
    (N_0, 1), lb[1], ub[1], dtype="float32")  # inital condition
X_0 = tf.concat([t_0, x_0], axis=1)

# Evaluate intitial condition at x_0
# fun_u is the function of interest, fun_u_0 defines boundary condition
u_0 = fun_u_0(x_0)

# Boundary data
t_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype="float32")
x_b = lb[1] + (ub[1] - lb[1]) * \
    tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype="float32")
X_b = tf.concat([t_b, x_b], axis=1)

# Evaluate boundary condition at (t_b,x_b)
u_b = fun_u_b(t_b, x_b)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype="float32")
x_r = tf.random.uniform((N_r, 1), lb[1], ub[1], dtype="float32")
X_r = tf.concat([t_r, x_r], axis=1)

# Collect boundary and inital data in lists
# we loop over these in the loss function to compute boundary conditions --> this enforces the physics?
X_data = [X_0, X_b]
u_data = [u_0, u_b]

# Initialize model
model = PINN(lb, ub)
model.build(input_shape=(None, 2))

# Initilize PINN solver
solver = PINNSolver(model, X_r)

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [1000, 3000], [1e-2, 1e-3, 5e-4])
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve(optim, X_data, u_data, N=400)

solver.plot_solution()
solver.plot_loss_history()
