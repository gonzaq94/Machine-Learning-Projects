from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np


def data_split(X, y, test_size):
    """

    :param X: input data
    :param y: input labels
    :param test_size: percentage of dataset used for test
    :return: the splitted data
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(model, x, y, epochs=50):
    """

    :param model: the model that will be used for training
    :param x: input data
    :param y: input labels
    :param epochs: number of epochs for training
    :return: the trained model
    """
    if isinstance(model, Sequential):

        model.fit(x, y, epochs=epochs, verbose=0)

    else:
        model.fit(x, y)

    return model


def validation(model, x, y):
    """

    :param model: instance of a model
    :param x: input data
    :param y: input labels
    :return: the accuracy score obtained
    """
    if isinstance(model, Sequential):

        y_pred = model.predict(x)

        y_pred = np.array([y_pred > 0.5]).astype(np.int16)

        y_pred = y_pred.reshape(len(y))

        return accuracy_score(y, y_pred)

    else:

        return model.score(x, y)


def plot_confusion_matrix(model, x, y_true, model_string, cm_labels, train_flag, dataset, image_folders="Output/Images"):
    """

    :param model: instance of the model
    :param x: input data
    :param y_true: true labels
    :param model_string: name of model
    :param cm_labels: type of colormap
    :param train_flag: true if train, else false
    :param dataset: dataset used,for naming the image
    :param image_folders: folder to save images
    :return:
    """
    y_pred = model.predict(x)

    if isinstance(model, Sequential):
        y_pred = np.array([y_pred >= 0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y_true))

    cm = confusion_matrix(y_true, y_pred)

    # we normalize the confusion matrix
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    # we plot

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    fmt = '.2f'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if train_flag:
        title = model_string + ": confusion matrix for train set"
        fig_name = model_string + '-' + dataset + '-cm-train.png'
    else:
        title = model_string + ": confusion matrix for test set"
        fig_name = model_string + '-' + dataset + '-cm-test.png'

    labels = ['', '', cm_labels[0], '', '', '', cm_labels[1]]

    ax.set(  # xticks=np.arange(cm.shape[1]),
        # yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=labels, yticklabels=labels,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()

    plt.savefig(image_folders + "/" + fig_name)
    plt.close()

def create_accs_file(filepath, dataset):
    """

    :param filepath: accuracy file location
    :param dataset: dataset name
    :return: file
    """

    accs_file = open(filepath, "w+")

    accs_file.write("Accuracy file for: " + dataset + " dataset" + "\n")

    return accs_file


def print_acc_2_file(file, model, accuracy):
    """

    :param file: file where to write
    :param model: model name
    :param accuracy: accuracy obtained with that model
    :return:
    """

    acc_str = '%.3f' % accuracy

    file.write(models_string_dic[model] + ": " + acc_str +"\n")


# Dictionary used to set the image title more properly

models_string_dic = {
    'decision_tree_model': 'Decision_tree',
    'knn_model': 'KNN',
    'svm_model': 'SVM',
    'neural_network': 'Neural_network',
    'bayes_model': 'Bayes_classifier'
}