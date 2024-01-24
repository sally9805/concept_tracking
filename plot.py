import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

def read_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def perform_pca(data_dict, num_components=2):
    # Extracting the vectors and creating a matrix
    data_matrix = np.array(list(data_dict.values()))

    # Standardizing the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_matrix)

    # Applying PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(standardized_data)

    # Returning the result
    return principal_components


def plot_vectors(strings, vectors, special_colors={'$1': 'red', '$5': 'blue', '$10': 'green'}):
    """
    Plots 2D vectors with specified strings, coloring specific strings differently.

    :param strings: List of strings corresponding to each vector.
    :param vectors: List of 2D vectors.
    :param special_colors: Dictionary mapping specific strings to colors.
    """
    for string, vector in zip(strings, vectors):
        color = special_colors.get(string, 'gray')  # Default color is gray
        plt.scatter(vector[0], vector[1], color=color)
        plt.text(vector[0], vector[1], string, fontsize=9)

        # To avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    plt.legend(*zip(*unique), loc='upper left')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('1900s')
    plt.show()


if __name__ == '__main__':
    dict1 = read_dict_from_pickle('1900s_anchor_embedding.pkl')
    dict2 = read_dict_from_pickle('1900s_adjectives_embedding.pkl')
    dict2.update(dict1)
    dict = {key: value for key, value in dict2.items() if not all(v == 0 for v in value)}
    result = perform_pca(dict)
    plot_vectors(dict.keys(), result)


