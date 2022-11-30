# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
from pinn import PINN
from pinn_solver import PINNSolver
from data import get_data

tf.keras.backend.set_floatx("float32")
tf.random.set_seed(0)

lower_bounds, upper_bounds, residual_sample_points, sample_points, points = get_data()

model = PINN(lower_bounds, upper_bounds)
model.build(input_shape=(None, 2))

solver = PINNSolver(model, residual_sample_points)

learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [1000, 3000], [1e-2, 1e-3, 5e-4])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
solver.solve(optimizer, sample_points, points, N=400)

solver.plot_solution()
solver.plot_loss_history()
