from models import *
from utilities import *
from prepro import *
import argparse
import sys

if __name__ == "__main__":

    # Some general parameters

    output_folder = "Output"
    accs_file_path = output_folder + "/accuracies_file"
    test_size = 0.33

    # Dataset and models to use are passed by command

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='the list of models you want to use',
                        default='svm_model')

    parser.add_argument('--dataset', type=str, default='kidney-disease',
                        help='Dataset to use: either kidney-disease or '
                             'bank-note ')
    parser.add_argument('--do_pca', help='If you want ot perform PCA', action='store_true')
    args = parser.parse_args()

    models_string = [item for item in args.models.split(',')]
    if args.dataset == 'kidney-disease':
        path = "data_classification/kidney_disease.csv"
        accs_file_path = accs_file_path + '_kidney_disease.txt'
        X, y, label, mod = load_preprocessing_data(path, index_col=0, binar=True)  # for kidney_disease
    elif args.dataset == 'bank-note':
        path = "data_classification/data_banknote_authentication.txt"
        accs_file_path = accs_file_path + '_bank_note.txt'
        X, y, label, mod = load_preprocessing_data(path, header=None, binar=True)  # for banknote
    else:
        print('dataset not available or misspelled')
        sys.exit(1)

    y_labels = [label[0], label[1]]
    if args.do_pca:
        DO_PCA = True
        if args.dataset == 'kidney-disease':
            X = pca(X, 'kidney-disease', mod)
        else:
            X = pca(X, 'bank-note', mod)
    else:
        DO_PCA = False

    # load preprocessed data

    X_train, X_test, y_train, y_test = data_split(X, y, test_size)
    if args.dataset == 'kidney-disease':
        models = call_models(models_string, 'kidney-disease', DO_PCA)
    else:
        models = call_models(models_string, 'bank-note', DO_PCA)

    # create the output files

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder + '/Images')

    accs_file = create_accs_file(accs_file_path, args.dataset)

    for i, model in enumerate(models):
        train_model(model, X_train, np.ravel(y_train))

        accuracy = validation(model, X_test, y_test)

        print(models_string[i], ' accuracy: ', '%.3f' % accuracy)

        print_acc_2_file(accs_file, models_string[i], accuracy)

        # we plot the confusion matrix for both the train and test datasets

        plot_confusion_matrix(model, X_train, y_train, models_string_dic[models_string[i]], y_labels, train_flag=True,
                              dataset=args.dataset)

        plot_confusion_matrix(model, X_test, y_test, models_string_dic[models_string[i]], y_labels, train_flag=False,
                              dataset=args.dataset)
