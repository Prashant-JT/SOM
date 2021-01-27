import tensorflow as tf
import numpy as np


class SOM:

    def __init__(self, m, n, input_dim, lr=0.3, sigma=None):

        self.m = m
        self.n = n
        self.input_dim = input_dim
        self.alpha = np.float(lr)  # Tasa de aprendizaje

        # Radio de afectación a las neuronas vecinas
        if sigma is None:
            self.sigma = max(m, n) / 1.5
        else:
            self.sigma = np.float(sigma)

        # Se inicilizan todos los pesos de las neuronas aleatoriamente,
        # las cuales son almacenadas en una matriz tipo "Variable" [m*n, input_dim]
        # donde "input_dim" es igual al número de características, y
        # "m" y "n" son las dimensiones del mapa auto-organizado.
        self.weights = tf.Variable(
            tf.random.uniform(shape=[self.m * self.n, self.input_dim]))

        # Variable que almacena las pocisiones de las neuronas
        self.locations = tf.constant(np.array(list(self._neuron_locations())))

    def _neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])  # Generador que devuelve la posición de cada neurona

    def get_weights(self):
        return self.weights  # Devuelve los pesos finales de las neuronas

    def get_bmus_locations(self):
        return self.bmu_locations  # Devuelve las localizaciones de la mejor neurona

    def closers(self, vect_input):
        # Se trata de la función que calcula la neurona más similar a los datos de entrada. sum((w-i)^2)
        self.vect_input = vect_input
        self.sqrt_distances = tf.reduce_sum(
            tf.pow(tf.subtract(tf.expand_dims(self.weights, axis=0), tf.expand_dims(self.vect_input, axis=1)), 2), 2)
        self.bmu_index = tf.argmin(self.sqrt_distances, axis=1)
        self.bmu_locations = tf.reshape(tf.gather(self.locations, self.bmu_index), [-1, 2])

    # Actualización de los pesos
    def updates(self, iter_input, num_epoch):

        update_aux = tf.multiply(iter_input,
                                 tf.divide(tf.cast(tf.subtract(self.alpha, 1), tf.float32),
                                           tf.cast(tf.subtract(num_epoch, 1), tf.float32)))

        self.sigma = tf.subtract(self.sigma, update_aux)  # Actualización de sigma (radius)
        self.alpha = tf.subtract(self.alpha, update_aux)  # Actualización de alpha (learning rate)

        sqrt_bmu_distances = tf.reduce_sum(
            tf.pow(tf.subtract(
                tf.expand_dims(self.locations, axis=0),
                tf.expand_dims(self.bmu_locations, axis=1)), 2),
            2)

        neighbourhood_eq = tf.exp(tf.divide(tf.negative(tf.cast(
            sqrt_bmu_distances, "float32")), tf.multiply(
            tf.square(tf.multiply(self.sigma, 0.08)), 2)))

        learning_rate_op = tf.multiply(neighbourhood_eq, self.alpha)

        numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(learning_rate_op, axis=-1),
                        tf.expand_dims(self.vect_input, axis=1)), axis=0)

        denominator = tf.expand_dims(
            tf.reduce_sum(learning_rate_op, axis=0) + float(1e-20), axis=-1)

        new_weights = tf.divide(numerator, denominator)
        return tf.compat.v1.assign(self.weights, new_weights)

    def train(self, v_data, num_epochs, data_size):

        vec_input = tf.compat.v1.placeholder(shape=[data_size, self.input_dim], dtype=tf.float32)
        iter_input = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)

        self.closers(vec_input)
        weights_update = self.updates(iter_input, num_epochs)

        with tf.compat.v1.Session() as session:

            session.run(tf.compat.v1.global_variables_initializer())

            # Comienza el entrenamiento
            for n_epoch in range(num_epochs):
                session.run(weights_update, feed_dict={vec_input: v_data, iter_input: n_epoch})

                print("Epoca ----> ", n_epoch, " de ", num_epochs)

            # Obtiene el mapa de pesos y se normalizan
            weights = session.run(self.get_weights()).reshape(self.m, self.n, self.input_dim)
            for idx in range(0, self.input_dim):
                weights[:, :, idx] = (weights[:, :, idx] - weights[:, :, idx].min()) / (
                        weights[:, :, idx].max() - weights[:, :, idx].min())

            # Finalmente se obtiene las posiciones del vector más próximo
            locations = session.run(self.get_bmus_locations(), feed_dict={vec_input: v_data})

        return weights, locations
