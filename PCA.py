import numpy
import tensorflow

class PCA(object):
    """Class PCA

       Principal Component Analysis

    """
    def __init__(self, data):
        self.data = data

    def reduce(self, percentage_variability_explained):
        graph = tensorflow.Graph()
        with graph.as_default():
            X = tensorflow.compat.v1.placeholder(tensorflow.float32, shape=self.data.shape)
            singular_values, U, _ = tensorflow.linalg.svd(X)
            sigma = tensorflow.linalg.diag(singular_values)

        with tensorflow.compat.v1.Session(graph=graph) as session:
            self.U, self.singular_values, self.sigma = session.run([U, singular_values, sigma], feed_dict={X: self.data})

        normalized_singular_values = self.singular_values / sum(self.singular_values)
        ladder = numpy.cumsum(normalized_singular_values)
        n_dimensions = next(idx for idx, value in enumerate(ladder) if value >= percentage_variability_explained) + 1;

        with graph.as_default():
            sigma = tensorflow.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])
            pca = tensorflow.matmul(self.U, sigma)

        with tensorflow.compat.v1.Session(graph=graph) as session:
            return session.run(pca, feed_dict={X: self.data})
