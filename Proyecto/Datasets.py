import numpy as np
import pandas as pd
from Process import process_data

path_crime = 'Criminalidad_Chicago_2018/Criminalidad_Chicago_2018.csv'
path_iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


class Datasets:
    def __init__(self, batch_data=1, column=None, to_numpy=True):
        self.batch_data = batch_data
        self.column = column
        self.to_numpy = to_numpy
        self.default = 'colors'
        self.datasets = {'colors': self._get_colors,
                         'iris': self._get_iris,
                         'crime': self._get_crime}

    def get_dataset(self, which='colors'):

        if which is 'crime' and self.column is None:
            self.column = 'Primary Type'  # Columna por defecto para dataset "crime"

        if which is not 'crime' and self.column is not None:
            return None, None

        if which not in self.datasets:
            return self.datasets[self.default]()

        return self.datasets[which]()

    def _get_colors(self):
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

        return colors, color_names

    def _get_iris(self):
        df = pd.read_csv(path_iris, header=None)
        df = df.sample(frac=self.batch_data)

        if self.to_numpy:
            data = df.iloc[:, 0:4].values
        else:
            data = df

        data_h = df.iloc[:, 4]

        return data, data_h

    def _get_crime(self):
        df, df_dict = process_data(path_crime)
        df = df.sample(frac=self.batch_data)

        if self.to_numpy:
            v_data = df.iloc[:, :].values
        else:
            v_data = df

        v_header = df[self.column].values

        return v_data, v_header

