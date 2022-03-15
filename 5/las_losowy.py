# Patryk Bandyra

from math import sqrt
from random import randrange, seed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sn
import pandas as pd

'''
Classes:
    1 - Kama
    2 - Rosa
    3 - Canadian
Attributes:
    1. area A,
    2. perimeter P,
    3. compactness C = 4*pi*A/P^2,
    4. length of kernel,
    5. width of kernel,
    6. asymmetry coefficient
    7. length of kernel groove.
'''


def load_data(filename):
    """
    Loads data into list of lists and convert string numbers to floats
    :param filename: name of data file
    :return: dataset (list of lists)
    """
    with open(filename, 'r', encoding='UTF-8') as file:
        data = file.readlines()
    return_set = []
    for line in data:
        temp = []
        for val in line.split():
            try:
                temp.append(float(val))
            except ValueError:
                temp.append(val.replace('\n', ''))
        return_set.append(temp)
    float_column_to_int(return_set)
    return return_set


def float_column_to_int(dataset):
    """
    Turns class column into integer
    :param dataset: list of lists
    :return: modified dataset
    """
    for observation in dataset:
        observation[-1] = int(observation[-1])
    return dataset


def cross_validation_split(dataset, n_folds):
    """
    Splits dataset into k folds choosing random items
    :param dataset: list of lists
    :param n_folds: number of folds
    :return: split dataset into n folds (list of lists)
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_accuracy_metric(actual, predicted):
    """
    Calculates accuracy percentage
    :param actual: list of actual values
    :param predicted: list of predicted values
    :return: percentage of accuracy
    """
    correct = 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            correct += 1
    return correct / float(len(actual)) * 100


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Evaluates an algorithm using a cross validation split
    :param dataset: list of lists
    :param algorithm: function
    :param n_folds: number of folds to cross validation
    :param args:
    :return: scores
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    all_predicted = list()
    all_actual = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        all_predicted += predicted
        print(f'Predicted: {predicted}')
        actual = [row[-1] for row in fold]
        all_actual += actual
        print(f'Actual:    {actual}')
        accuracy = calculate_accuracy_metric(actual, predicted)
        scores.append(accuracy)
    # confusion matrix
    matrix = confusion_matrix(all_actual, all_predicted)

    print('Confusion matrix measures')
    print(f'Accuracy: {accuracy_score(all_actual, all_predicted)}')
    print(f'Micro Precision: {precision_score(all_actual, all_predicted, average="micro")}')
    print(f'Micro Recall: {recall_score(all_actual, all_predicted, average="micro")}')
    print(f'Micro F1-score: {f1_score(all_actual, all_predicted, average="micro")}\n')

    print(f'Macro Precision: {precision_score(all_actual, all_predicted, average="macro")}')
    print(f'Macro Recall: {recall_score(all_actual, all_predicted, average="macro")}')
    print(f'Macro F1-score: {f1_score(all_actual, all_predicted, average="macro")}\n')

    print(f'Weighted Precision: {precision_score(all_actual, all_predicted, average="weighted")}')
    print(f'Weighted Recall: {recall_score(all_actual, all_predicted, average="weighted")}')
    print(f'Weighted F1-score: {f1_score(all_actual, all_predicted, average="weighted")}\n')

    print('Classification Report\n')
    print(classification_report(all_actual, all_predicted, target_names=['1', '2', '3']))

    df_cm = pd.DataFrame(matrix, range(3), range(3))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()
    return scores


def split_database_on_attribute(index, value, dataset):
    """
    Splits a dataset based on an attribute and an attribute value
    :param index: index of a column
    :param value: value of an attribute
    :param dataset: list of lists
    :return: split dataset - 2 list of lists
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, classes):
    """
    Calculates the Gini index for a split dataset
    :param groups: list of lists
    :param classes: list of classes
    :return: Gini index (number)
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0
    for group in groups:
        size = float(len(group))    # group size
        # avoid divide by zero
        if size == 0:
            continue
        score = 0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1 - score) * (size / n_instances)
        return gini


def get_split(dataset, n_features):
    """
    Selects the best split point for a dataset
    :param dataset: list of lists
    :param n_features: number of attributes in each observation
    :return: dictionary - index, value, groups
    """
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = split_database_on_attribute(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


def to_terminal(group):
    """
    Creates a terminal node value
    :param group:
    :return: terminal node value
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
    """
    Creates child splits for a node or makes terminal
    :param node:
    :param max_depth:
    :param min_size:
    :param n_features:
    :param depth:
    :return:
    """
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


def build_tree(train, max_depth, min_size, n_features):
    """
    Builds decision tree
    :param train: train dataset (list fo lists)
    :param max_depth: number
    :param min_size: number
    :param n_features: number
    :return: root of a tree
    """
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    """
    Makes a prediction with a decision tree
    :param node:
    :param row:
    :return: predicted class
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def subsample(dataset, ratio):
    """
    Creates a random subsample from the dataset with replacement
    :param dataset:
    :param ratio:
    :return:
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def bagging_predict(trees, row):
    """
    Makes a prediction with a list of bagged trees
    :param trees: list of trees
    :param row:
    :return:
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Performs Random Forest Algorithm
    :param train: train dataset
    :param test: test dataset
    :param max_depth:
    :param min_size:
    :param sample_size:
    :param n_trees:
    :param n_features:
    :return: predictions
    """
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        # print(tree)   # uncomment to display tree (dictionary)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


def display_tree():
    pass


def run():
    """
    Runs algorithm and prints accuracy results - number of trees dependency.
    """
    seed(5)     # random seed
    dataset = load_data('seeds_dataset.txt')
    # Constant values
    n_folds = 5     # k value of cross split validation
    max_depth = 3
    min_size = 1
    sample_size = 1
    n_features = int(sqrt(len(dataset[0]) - 1))
    for n_trees in [1, 5, 10, 15, 20, 25, 30, 35]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print(f'Trees: {n_trees}')
        print(f'Scores: {scores}')
        print(f'Mean Accuracy: {(sum(scores) / float(len(scores)))}')
        print('-------------------------------------------------------------------------------------------------------')


def run2():
    """
    Runs algorithm and prints accuracy results - depth dependency.
    """
    seed(5)     # random seed
    dataset = load_data('seeds_dataset.txt')
    # Constant values
    n_folds = 5     # k value of cross split validation
    min_size = 1
    sample_size = 1
    n_features = int(sqrt(len(dataset[0]) - 1))
    n_trees = 5
    for max_depth in [1, 5, 10, 15, 20, 25]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print(f'Max depth: {max_depth}')
        print(f'Scores: {scores}')
        print(f'Mean Accuracy: {(sum(scores) / float(len(scores)))}')
        print('-------------------------------------------------------------------------------------------------------')


def run3():
    """
    Runs algorithm and prints accuracy results - k value dependency (cross validation parameter).
    """
    seed(5)     # random seed
    dataset = load_data('seeds_dataset.txt')
    # Constant values
    max_depth = 10
    min_size = 1
    sample_size = 1
    n_features = int(sqrt(len(dataset[0]) - 1))
    n_trees = 5
    for n_folds in [5, 10, 15, 20, 25, 30, 35]:    # k value of cross split validation
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print(f'K value: {n_folds}')
        print(f'Scores: {scores}')
        print(f'Mean Accuracy: {(sum(scores) / float(len(scores)))}')
        print('-------------------------------------------------------------------------------------------------------')


def print_graph(dataset, row_x, row_y, row_x_info, row_y_info):
    """
    Displays graph for a specific dataset
    :param dataset: list of lists
    :param row_x: number of attribute
    :param row_y: number of attribute
    :param row_x_info: string
    :param row_y_info: string
    """
    x_values = []
    y_values = []
    colors = []
    for data_row in dataset:
        x_values.append(data_row[row_x])
        y_values.append(data_row[row_y])
        if data_row[-1] == 1:
            colors.append('red')
        elif data_row[-1] == 2:
            colors.append('blue')
        else:
            colors.append('green')
    for x, y, col in zip(x_values, y_values, colors):
        plt.scatter(x, y, c=col, s=5, linewidth=0)

    red_p = mpatches.Patch(color='red', label='Kama')
    blue_p = mpatches.Patch(color='blue', label='Rosa')
    green_p = mpatches.Patch(color='green', label='Canadian')
    plt.legend(handles=[red_p, blue_p, green_p], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel(row_x_info)
    plt.ylabel(row_y_info)
    plt.grid()
    plt.show()


def analyze_dataset():
    """
    Analyzes all possible attribute configurations
    """
    dataset = load_data('seeds_dataset.txt')
    print_graph(dataset, 0, 1, 'area', 'perimeter')
    print_graph(dataset, 0, 2, 'area', 'compactness')
    print_graph(dataset, 0, 3, 'area', 'length of kernel')
    print_graph(dataset, 0, 4, 'area', 'width of kernel')
    print_graph(dataset, 0, 5, 'area', 'asymmetry coefficient')
    print_graph(dataset, 0, 6, 'area', 'length of kernel groove')
    print_graph(dataset, 1, 2, 'perimeter', 'compactness')
    print_graph(dataset, 1, 3, 'perimeter', 'length of kernel')
    print_graph(dataset, 1, 4, 'perimeter', 'width of kernel')
    print_graph(dataset, 1, 5, 'perimeter', 'asymmetry coefficient')
    print_graph(dataset, 1, 6, 'perimeter', 'length of kernel groove')
    print_graph(dataset, 2, 3, 'compactness', 'length of kernel')
    print_graph(dataset, 2, 4, 'compactness', 'width of kernel')
    print_graph(dataset, 2, 5, 'compactness', 'asymmetry coefficient')
    print_graph(dataset, 2, 6, 'compactness', 'length of kernel groove')
    print_graph(dataset, 3, 4, 'length of kernel', 'width of kernel')
    print_graph(dataset, 3, 5, 'length of kernel', 'asymmetry coefficient')
    print_graph(dataset, 3, 6, 'length of kernel', 'length of kernel groove')
    print_graph(dataset, 4, 5, 'width of kernel', 'asymmetry coefficient')
    print_graph(dataset, 4, 6, 'width of kernel', 'length of kernel groove')
    print_graph(dataset, 5, 6, 'asymmetry coefficient', 'length of kernel groove')


if __name__ == '__main__':
   # analyze_dataset()
    run()
   # run2()
   # run3()