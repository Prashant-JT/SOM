import tensorflow as tf
import numpy as np


class SOM:

    def __init__(self, m, n, dim, learning_rate_som=0.3, radius_factor=1.3):

        self.m = m
        self.n = n
        self.dim = dim
        self.alpha = learning_rate_som
        self.sigma = max(m, n) * 2.0

        self.map = tf.Variable(tf.random.uniform(shape=[m * n, dim], minval=0, maxval=1, seed=2))
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))


    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self):
        return self.map

    def getlocation(self):
        return self.bmu_locs

    def feedforward(self, input):

        self.input = input
        self.squared_distance = tf.reduce_sum(
            tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0), tf.expand_dims(self.input, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self, iter, num_epoch):

        # Update the weigths
        radius = tf.subtract(self.sigma,
                             tf.multiply(iter,
                                         tf.divide(tf.cast(tf.subtract(self.alpha, 1), tf.float32),
                                                   tf.cast(tf.subtract(num_epoch, 1), tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                        tf.divide(tf.cast(tf.subtract(self.alpha, 1), tf.float32),
                                                  tf.cast(tf.subtract(num_epoch, 1), tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
            tf.pow(tf.subtract(
                tf.expand_dims(self.location_vects, axis=0),
                tf.expand_dims(self.bmu_locs, axis=1)), 2),
            2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
            self.bmu_distance_squares, "float32")), tf.multiply(
            tf.square(tf.multiply(radius, 0.08)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)

        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
                        tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op, axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.divide(self.numerator, self.denominator)
        self.update = tf.compat.v1.assign(self.map, self.new_weights)

        return self.update
