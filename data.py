import tensorflow as tf
import numpy as np


def ic(x):
    pi = tf.constant(np.pi, dtype="float32")
    return tf.sin(pi * x)


def bc(t, x):
    n = x.shape[0]
    return tf.zeros((n, 1), dtype="float32")


def residual(t, x, u, u_t, u_x, u_xx):
    viscosity = 5
    return u_t + u * u_x - viscosity * u_xx


def get_data():
    N_ic = 50
    N_bc = 50
    N_residual = 10000

    min_t = 0.
    max_t = 1.
    min_x = 0.
    max_x = 1.

    lower_bounds = tf.constant([min_t, min_x], dtype="float32")
    upper_bounds = tf.constant([max_t, max_x], dtype="float32")

    t_ic_sample_points = tf.ones((N_ic, 1), dtype="float32") * lower_bounds[0]
    x_ic_sample_points = tf.random.uniform(
        (N_ic, 1), lower_bounds[1], upper_bounds[1], dtype="float32")
    t_and_x_ic_sample_points = tf.concat(
        [t_ic_sample_points, x_ic_sample_points], axis=1)

    ic_points = ic(x_ic_sample_points)

    t_bc_sample_points = tf.random.uniform((N_bc, 1),
                                           lower_bounds[0], upper_bounds[0], dtype="float32")
    x_bc_sample_points = lower_bounds[1] + (upper_bounds[1] - lower_bounds[1]) * \
        tf.keras.backend.random_bernoulli(
            (N_bc, 1), 0.5, dtype="float32")
    t_and_x_bc_sample_points = tf.concat(
        [t_bc_sample_points, x_bc_sample_points], axis=1)

    bc_points = bc(t_bc_sample_points, x_bc_sample_points)

    t_residual_sample_points = tf.random.uniform(
        (N_residual, 1), lower_bounds[0], upper_bounds[0], dtype="float32")
    x_residual_sample_points = tf.random.uniform(
        (N_residual, 1), lower_bounds[1], upper_bounds[1], dtype="float32")
    t_and_x_residual_sample_points = tf.concat(
        [t_residual_sample_points, x_residual_sample_points], axis=1)

    ic_and_bc_sample_points = [
        t_and_x_ic_sample_points, t_and_x_bc_sample_points]
    ic_and_bc_points = [ic_points, bc_points]

    return lower_bounds, upper_bounds, t_and_x_residual_sample_points, ic_and_bc_sample_points, ic_and_bc_points
