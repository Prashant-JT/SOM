import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PCA
from Process import process_data

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
