# Title :
# Author :yuqian
# Email :760203432@qq.com
# Time :2023/2/10

import os
import warnings
import seaborn as sns
import numpy as np
import pandas as pd

from codes.Data_Augmentation.Get_data import drop_cols_minmax, read_dataset
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model

from codes.Feature_Selection.autoencoder import read_dataset_standardscaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style('white')


def reset_index1(X, y, name):
    temp = X[y['type'] == name]
    temp.reset_index(drop=True, inplace=True)
    values = temp.values
    return values


def relu(x):
    """rectified linear unit (activation function)"""
    return max(0, x)


def get_influence(input, output):
    """getting gene influence (%) score and indices list when the inputs are reversely sorted"""
    influence = []
    for i in input:
        influence.append(i / output)
    influence = np.asarray(influence) * 100
    rev_sorted_idx = sorted(range(len(influence)), key=lambda k: influence[k], reverse=True)
    return influence, rev_sorted_idx


def count_and_sort_genes(genes):
    unique, counts = np.unique(genes, return_counts=True)
    count_dict = dict(zip(unique, counts))
    genes_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    return genes_counts


def get_wighted_influence(relu_output_list, influencial_idx_list):
    all_idx = np.hstack(influencial_idx_list)
    unique, counts = np.unique(all_idx, return_counts=True)
    count_dict = dict(zip(unique, counts))

    relu_output_inf_percentage = relu_output_list / sum(relu_output_list)
    idx_per_list = np.ceil(relu_output_inf_percentage * INF_IDX_LIMIT)
    idx_per_list = np.asarray(idx_per_list, dtype=int)

    reserved_inf_by_percentage = []
    for i in range(len(influencial_idx_list)):
        idx_per_list_limit = idx_per_list[i]  # mostly 1, could be 2
        reserved_inf_by_percentage.extend(influencial_idx_list[i][:idx_per_list_limit])
    all_selected_idx = np.unique(reserved_inf_by_percentage)
    print(f'# total filtered_genes: {len(all_selected_idx)}')
    return np.asarray(all_selected_idx)


def get_all_influencers_gene_names(gene_names, all_influencers_idx):
    influencer_gene_names = []
    for i in all_influencers_idx:
        influencer_gene_names.append(gene_names[int(i)])
    return np.asarray(influencer_gene_names)


if __name__ == '__main__':

    layer1 = 250
    layer2 = 21
    path = "../../dataset/dataset_nice_classification/"
    X, y = read_dataset_standardscaler(path, "")
    le = LabelEncoder()
    num_label = le.fit_transform(y.values)
    y['Class'] = num_label

    data = y['Class'].value_counts()
    hist = pd.Series(data)
    classes = [le.classes_[i] for i in hist.index]

    # X, deleted_genes = drop_cols_minmax(X)
    gene_names = np.asarray(X.columns)
    IO_DIM = X.shape[1]
    model_names = [c + '_ae' for c in classes]
    all_values = []
    for c in classes:
        all_values.append(reset_index1(X, y, c))

    models = []
    for model in model_names:
        ae = load_model(f"{path}{model}.h5")
        models.append(ae)

    for class_idx in range(len(model_names)):
        model = models[class_idx]
        data = all_values[class_idx]
        data_class = classes[class_idx]
        layer_weights = model.get_weights()

        gene_names_from_all_input = []
        INF_IDX_LIMIT = layer2

        print(f'\n[{class_idx + 1}/{len(classes)}] filtering for {data_class}')
        print('=========================================================')
        for j in range(data.shape[0]):
            relu_output_list = []
            influencial_idx_list = []
            secend_layer_input = []
            for i in range(layer1):
                input_x_weight = layer_weights[0][:, i] * data[j]
                neural_output = sum(input_x_weight) + layer_weights[1][i]
                if relu(neural_output):
                    _, influence_idx = get_influence(input_x_weight, relu(neural_output))
                    relu_output_list.append(relu(neural_output))
                    influencial_idx_list.append(influence_idx[:INF_IDX_LIMIT])
            relu_output_list = np.asarray(relu_output_list)
            influencial_idx_list = np.asarray(influencial_idx_list)
            all_influencers_idx = get_wighted_influence(relu_output_list, influencial_idx_list)
            influencer_gene_names = get_all_influencers_gene_names(gene_names, all_influencers_idx)
            gene_names_from_all_input.append(influencer_gene_names)

        gene_list = np.asarray(gene_names_from_all_input)
        all_genes = []
        for gene in gene_list:
            all_genes.extend(gene)
        all_genes_counts = count_and_sort_genes(all_genes)
        count_df = pd.DataFrame.from_dict(all_genes_counts)
        count_df.columns = ['gene', 'count']
        count_df.to_csv(f'{path}{data_class}_all_genes_counts.csv', index=False)

