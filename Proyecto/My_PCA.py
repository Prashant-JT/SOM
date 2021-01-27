import tensorflow as tf


class My_PCA:
    def __init__(self, data):
        self.eigen_vectors = None
        self.eigen_values = None
        self._center_and_cov(tf.cast(data, tf.float32))

    def _center_and_cov(self, data):
        self.data_centered = tf.identity(data) - tf.reduce_mean(data, axis=0)

        self.data_cov = tf.divide(
                            tf.tensordot(tf.transpose(self.data_centered), self.data_centered, axes=1),
                            tf.cast(tf.shape(data)[0], tf.float32)
                        )

    def get_percentage(self):
        return (self.eigen_values / tf.reduce_sum(self.eigen_values)).numpy()

    def compute_PCA(self):
        # Encuentra autovectores y autovalores
        self.eigen_values, self.eigen_vectors = tf.linalg.eigh(self.data_cov)

        # Se reordenan los autovalores en forma decreciente, al igual que los autovectores
        sorted_index = tf.argsort(self.eigen_values)[::-1]
        self.eigen_values = tf.gather(self.eigen_values, sorted_index)
        self.eigen_vectors = tf.gather(self.eigen_vectors, sorted_index, axis=1)

        return (tf.matmul(self.data_centered, self.eigen_vectors)).numpy()
