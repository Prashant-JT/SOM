import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PCA
from Process import process_data
from SOM import SOM
import tensorflow as tf

path = 'Criminalidad_Chicago_2018/Criminalidad_Chicago_2018.csv'


def test_0():
    v_data = np.array([[8.7, 0.3, 28.1],
                          [14.3, 0.9, 32.4],
                          [18.9, 1.8, 34.0],
                          [19.0, 0.8, 34.4],
                          [20.5, 0.9, 33.3],
                          [14.7, 1.1, 32.6],
                          [18.8, 2.5, 37.6],
                          [37.3, 2.7, 43.1],
                          [12.6, 1.3, 30.9],
                          [25.7, 3.4, 40.9]])

    pca = PCA.PCA(v_data)
    v_data_reduced = pca.reduce(percentage_variability_explained=0.9)

    print(v_data_reduced)


def test_1():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print(df.head(3))
    print(df.tail(3))
    v_data = df.iloc[0:, 0:4].values
    print(v_data.shape)

    matplotlib.pyplot.scatter(v_data[0:50, 0], v_data[0:50, 1], color='gray', marker='o', label='setosa')
    matplotlib.pyplot.scatter(v_data[50:100, 0], v_data[50:100, 1], color='cyan', marker='x',
                              label='virginica')
    matplotlib.pyplot.scatter(v_data[100:150, 0], v_data[100:150, 1], color='yellow', marker='*',
                              label='versicolor')

    pca = PCA.PCA(v_data)
    v_data_reduced = pca.reduce(percentage_variability_explained=0.9)

    print(v_data_reduced.shape)

    matplotlib.pyplot.scatter(v_data_reduced[0:50, 0], v_data_reduced[0:50, 1], color='red', marker='o', label='setosa')
    matplotlib.pyplot.scatter(v_data_reduced[50:100, 0], v_data_reduced[50:100, 1], color='blue', marker='x', label='virginica')
    matplotlib.pyplot.scatter(v_data_reduced[100:150, 0], v_data_reduced[100:150, 1], color='black', marker='*', label='versicolor')

    matplotlib.pyplot.xlabel('1er Component')
    matplotlib.pyplot.ylabel('2da Component')
    matplotlib.pyplot.legend(loc='upper left')
    matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig('images/PCA.svg', format='svg')


def test_2():
    df, df_dict = process_data(path)
    v_data = df.iloc[:, 3:].values

    print(df.columns)

    print(v_data.shape)

    pca = PCA.PCA(v_data)
    v_data_reduced = pca.reduce(percentage_variability_explained=0.99)

    print(v_data_reduced.shape)


def test_3():
    # Training inputs for RGBcolors
    colors = np.array([[0., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 0.5],
                       [0.125, 0.529, 1.0],
                       [0.33, 0.4, 0.67],
                       [0.6, 0.5, 1.0],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 1., 1.],
                       [1., 0., 1.],
                       [1., 1., 0.],
                       [1., 1., 1.],
                       [.33, .33, .33],
                       [.5, .5, .5],
                       [.66, .66, .66]])

    color_names = ['black', 'blue', 'darkblue', 'skyblue',
                   'greyblue', 'lilac', 'green', 'red',
                   'cyan', 'violet', 'yellow', 'white',
                   'darkgrey', 'mediumgrey', 'lightgrey']

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    v_data = df.iloc[:, 0:4].values
    v_header = df.iloc[:, 4]
    n_char = len(v_data[0])

    tf.compat.v1.disable_eager_execution()

    # hyper parameter
    dim = 150
    SOM_layer = SOM(dim, dim, n_char)
    num_epoch = 1000
    batch_size = len(v_data)

    # create the graph
    x = tf.compat.v1.placeholder(shape=[batch_size, n_char], dtype=tf.float32)
    current_iter = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)

    # graph
    SOM_layer.feedforward(x)
    map_update = SOM_layer.backprop(current_iter, num_epoch)

    # session
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        # start the training
        for iter in range(num_epoch):
            for current_train_index in range(0, len(v_data), batch_size):
                currren_train = v_data[current_train_index:current_train_index + batch_size]
                sess.run(map_update, feed_dict={x: currren_train, current_iter: iter})
            print("Epoca ----> ", iter)

        # get the trained map and normalize
        trained_map = sess.run(SOM_layer.getmap()).reshape(dim, dim, n_char)
        for idx in range(0, n_char):
            trained_map[:, :, idx] = (trained_map[:, :, idx] - trained_map[:, :, idx].min()) / (
                        trained_map[:, :, idx].max() - trained_map[:, :, idx].min())

        # after training is done get the closest vector
        locations = sess.run(SOM_layer.getlocation(), feed_dict={x: v_data})

        plt.imshow(trained_map.astype(float))
        for i, m in enumerate(locations):
            plt.text(m[1], m[0], v_header[i], ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, lw=0))
        plt.axis('off')
        plt.title('Color SOM')
        plt.show()
