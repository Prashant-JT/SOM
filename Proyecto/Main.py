import matplotlib.pyplot as plt
import tensorflow as tf
from SOM import SOM
from My_PCA import My_PCA
from Datasets import Datasets


def get_pca(v_data, range_cols):
    my_pca = My_PCA(v_data)
    pca_data = my_pca.compute_PCA()

    percentage_data = my_pca.get_percentage()
    print("Procentaje de datos representado con 3 componentes principales: ", sum(percentage_data[range_cols]) * 100)

    return pca_data[:, range_cols]  # Se coge las tres componentes principales


def get_SOM(v_data):
    # Hiper-parámetros
    m = 100
    n = 100
    n_char = len(v_data[0])  # Número de características
    lr = 0.3  # Tasa de aprendizaje
    radius = m / 1.5  # Radio de actualización para los pesos de vecinos cercanos
    num_epoch = 100  # Número de épocas de entrenamiento

    SOM_layer = SOM(m, n, n_char, lr, radius)

    # Retorna el mapa de pesos resultante, y posiciones
    return SOM_layer.train(v_data, num_epoch, len(v_data))


def main():
    # Se le puede pasar la columna a clasificar (sólo Crimen), el procentaje de muestras deseado,
    # y si se desea obtener los datos como dataset o numpy array (sólo Iris, Crimen)
    datasets = Datasets(batch_data=0.005, column='District')
    v_data, v_header = datasets.get_dataset('crime')

    v_data = get_pca(v_data, range(0, 3))  # Aplicar PCA sólo a Crime dataset

    tf.compat.v1.disable_eager_execution()  # Evita un error de compatibilidad de tensorflow

    som_map, som_locations = get_SOM(v_data)

    plt.imshow(som_map)
    for i, m in enumerate(som_locations):
        plt.text(m[1], m[0], v_header[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.axis('off')
    plt.title('Color SOM')
    plt.show()
