import tensorflow as tf


class PINN(tf.keras.Model):
    def __init__(self, lower_bounds, upper_bounds, output_dim=1, num_hidden_layers=8, hidden_layer_output_size=20, activation='tanh', kernel_initializer='glorot_normal', **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - lower_bounds) / (upper_bounds - lower_bounds) - 1.0)
        self.hidden_layers = [tf.keras.layers.Dense(hidden_layer_output_size,
                                                    activation=tf.keras.activations.get(
                                                        activation),
                                                    kernel_initializer=kernel_initializer)
                              for _ in range(self.num_hidden_layers)]
        self.ouput_layer = tf.keras.layers.Dense(output_dim)

    def call(self, X):
        output = self.scaling_layer(X)
        for i in range(self.num_hidden_layers):
            output = self.hidden_layers[i](output)
        return self.ouput_layer(output)
