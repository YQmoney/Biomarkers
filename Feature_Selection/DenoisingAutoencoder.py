# Title :
# Author :yuqian
# Email :760203432@qq.com
# Time :2023/2/10


import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tabulate import tabulate
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from codes.Feature_Selection.autoencoder import read_dataset_standardscaler

# CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85


def standardize(x, mean=None, std=None):
    """
    Shape x: (nb_samples, nb_vars)
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def read_dataset(path, name):
    print('reading data...\n', end=' ')
    X = pd.read_csv(f'{path}/{name}data.csv')
    X = X.drop(columns=['Unnamed: 0'])
    print(f'finished with shape: {X.shape} and type {type(X)}.')

    print('reading labels...\n', end=' ')
    y = pd.read_csv(f'{path}/{name}labels.csv')
    y = y.drop(columns=['Unnamed: 0'])
    y.columns = ['type']
    print(f'finished with shape: {y.shape} and type {type(y)}.')
    X.index = y.index
    return X, y

def drop_cols_minmax(df):
    n_cols = df.shape[1]
    print(f'> input shape :: {df.shape}')
    deleted_genes = list(df.columns[df.min() == df.max()])
    print('> deleting columns with no impact (min==max) ...', deleted_genes)
    df = df.loc[:, (df.max() != df.min())]
    print('> total deleted cols ::', n_cols - df.shape[1])
    print(f'> output shape :: {df.shape}')
    return df, deleted_genes

sns.set_style('white')

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# logging.basicConfig(
#     filename=f'../../dataset/results/logging_info/logging_{time.strftime("%Y-%m-%d", time.localtime())}.log',
#     level=logging.INFO)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, restore_best_weights=True)


def Model(layer1, layer2, lr=0.001, summary=True):
    """

    :param layer1:
    :param layer2:
    :param summary:
    :return:
    """
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer1, input_dim=IO_DIM, activation='relu'))
    model.add(tf.keras.layers.Dense(layer2, activation='relu'))
    model.add(tf.keras.layers.Dense(layer1, activation='relu'))
    model.add(tf.keras.layers.Dense(IO_DIM))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    if summary:
        model.summary()
    return model


def evaluate(model, X_class):
    """

    :param model:
    :param X_class:
    :return:
    """
    loss = []
    for x in X_class:
        loss.append(np.sqrt(metrics.mean_squared_error(model.predict(x), x)))
    return loss


def get_pred(df_test, models):
    """

    :param df_test:
    :param models:
    :return:
    """
    y_pred = []
    for i in range(len(df_test)):
        loss = []
        x = df_test[i].reshape(1, IO_DIM)
        for model in models:
            loss.append(np.sqrt(metrics.mean_squared_error(model.predict(x), x)))
        loss_idx = loss.index(min(loss))
        y_pred.append(hist.index.values[loss_idx])
    return y_pred


def print_list(lst):
    return [round(i, 4) for i in list(lst)]


def report_model_performance(mat):
    """

    :param mat:
    :return:
    """
    TP = np.diag(mat)
    FP = mat.sum(axis=0) - TP
    FN = mat.sum(axis=1) - TP
    TN = mat.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    F1S = (2 * PPV * TPR) / (PPV + TPR)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    measures = {}
    measures['TP'] = TP
    measures['TN'] = TN
    measures['FP'] = FP
    measures['FN'] = FN
    measures['Confusion Matrix'] = mat
    measures['Sensitivity / hit rate / recall / true positive rate'] = print_list(TPR)
    measures['Specificity / true negative rate'] = print_list(TNR)
    measures['Precision / positive predictive value'] = print_list(PPV)
    measures['Negative predictive value'] = print_list(NPV)
    measures['F-1 score'] = print_list(F1S)
    measures['False positive rate'] = print_list(FPR)
    measures['False negative rate'] = print_list(FNR)
    measures['False discovery rate'] = print_list(FDR)
    measures['Classwise Accuracy'] = print_list(ACC)
    measures['Overall Accuracy'] = print_list([accuracy_score(y_test, y_pred)])

    return measures


def print_measures(measures):
    for c in classes:
        print(c, end='\t')
    print('\n--------------------------------------')
    for k in measures:
        if k == 'Overall Accuracy':
            print('\n' + k, '::', measures[k][0])
        elif k == 'Confusion Matrix':
            continue
        else:
            s = ''
            for m in measures[k]:
                s += str(m) + '\t'
            print(s, '::', k)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


if __name__ == '__main__':
    layer1 = 256
    layer2 = 32
    path = "../../datasets/datasets/CVAE_WGAN_GP/"
    X, y = read_dataset(path, "no_normal_")

    le = LabelEncoder()
    num_label = le.fit_transform(y.values)
    y['Class'] = num_label
    y = y.drop(columns=['type'])
    data = y['Class'].value_counts()
    hist = pd.Series(data)
    classes = [le.classes_[i] for i in hist.index]
    model_names = [c + '_ae' for c in classes]
    num_classes = len(classes)
    count_label = [data[i] for i in hist.index]


    IO_DIM = X.shape[1]  # n_genes

    folds = 10
    skf = StratifiedKFold(n_splits=folds)
    skf.get_n_splits(X, y)

    y_test_list = []
    y_pred_list = []
    all_measures = []
    no_of_run = 1
    i = 0
    while i < no_of_run:
        performance_measures = []
        k = 0
        for train_index, test_index in skf.split(X, y):
            with tf.compat.v1.Session(config=config) as sess:
                print(f'> Go :: {i + 1} | Fold :: {k + 1}')
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                masks = []
                for c in hist.index.values:
                    mask = y['Class'] == c
                    masks.append(mask)

                X_all_class = [X_train[mask] for mask in masks]

                X_class = []
                for df in X_all_class:
                    X_class.append(df.values)

                X_test = X_test.values
                y_train = y_train.values.flatten()
                y_test = y_test.values.flatten()

                histories = []
                losses = []
                models = []
                for x in X_class:
                    x_train, x_test = train_test_split(x, test_size=0.25, random_state=2023)

                    # add noisy
                    noise_factor = 0.5
                    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=0.3,
                                                                              size=x_train.shape)
                    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=0.3,
                                                                            size=x_test.shape)

                    autoencoder = Model(layer1=layer1, layer2=layer2, summary=False)
                    err = autoencoder.fit(x_train_noisy,
                                          x_train,
                                          validation_data=(x_test_noisy, x_test),
                                          verbose=0,
                                          epochs=5000,
                                          batch_size=128,
                                          callbacks=[early_stopping]
                                          )
                    histories.append(err)
                    losses.append(evaluate(autoencoder, X_class))
                    models.append(autoencoder)

                y_pred = get_pred(X_test, models)
                y_test_list.append(y_test)
                y_pred_list.append(y_pred)
                mat = confusion_matrix(y_test, y_pred)
                print('Confusion Matrix')
                print('--------------------------------------')
                print(mat, '\n\n')
                print("losses:", "\n", tabulate(losses), "\n\n")
                measures = report_model_performance(mat)
                if measures['Overall Accuracy'][0] == 1.0:
                    saved_model = models
                    print(f"\nmodels saved @ accuracy {measures['Overall Accuracy'][0]}\n")

                    for idx, model in enumerate(saved_model):
                        mkdir(f'{path}')
                        model.save(f'{path}/{model_names[idx]}.h5')
                    print("Saved model to disk")

                print_measures(measures)
                performance_measures.append(measures)
                k += 1
                print('==================================================\n')
                sess.close()
        all_measures.append(performance_measures)
        # logging.info(performance_measures)
        i += 1
