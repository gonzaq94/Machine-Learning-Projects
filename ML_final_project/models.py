import keras
import sklearn
from keras.layers import Dense
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def svm_model():
    """
    :return: instance of the model
    """
    return svm.SVC(gamma='scale')


def knn_model():
    """
    :return: instance of the model
    """
    return KNeighborsClassifier(n_neighbors=5)


def decision_tree_model():
    """
    :return: instance of the model
    """
    return tree.DecisionTreeClassifier()


def neural_network(input_dim, n_neurons_per_layer=None, n_layers=3):
    """
    :return: instance of the model
    """
    if n_neurons_per_layer is None:
        n_neurons_per_layer = [5, 3, 2]
    if len(n_neurons_per_layer) != n_layers:
        print('number of layers of network not match number of neurons per layer')
    else:
        classifier = keras.Sequential()
        # Input layer
        classifier.add(
            Dense(n_neurons_per_layer[0], activation='relu', kernel_initializer='random_normal',
                  input_dim=input_dim))
        for n in n_neurons_per_layer[1:]:
            # Hidden Layers
            classifier.add(Dense(n, activation='tanh', kernel_initializer='random_normal'))

            # Output Layer
        classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        # Compiling the neural network
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier


def gaussian_bayes_model():
    """

    :return:Gaussian Naive Bayes (GaussianNB)
    """
    return GaussianNB()


def call_models(list_models, dataset, do_pca):
    """
    :param do_pca: if to do pca or not
    :param dataset: type of dataset
    :param: list_models: the names of the models you want to use
    :type list_models: list
    """
    list_of_models = list()
    for model in list_models:
        if type(model) == str:
            if model == 'svm_model':
                list_of_models.append(svm_model())
            elif model == 'knn_model':
                list_of_models.append(knn_model())
            elif model == 'decision_tree_model':
                list_of_models.append(decision_tree_model())
            elif model == 'bayes_model':
                list_of_models.append(gaussian_bayes_model())
            elif model == 'neural_network':
                if dataset == 'kidney-disease':
                    if do_pca:
                        input_dim = 10
                    else:
                        input_dim = 24
                else:
                    if do_pca:
                        input_dim = 2
                    else:
                        input_dim = 4
                try:
                    list_of_models.append(neural_network(input_dim))
                except:
                    print('Not loaded neural network')
                    continue
    return list_of_models
