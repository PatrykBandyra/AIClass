# Patryk Bandyra
from math import sqrt, exp, pi
from random import randrange, seed, shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_data(filename):
    '''
    Loads data into list of lists and convert string numbers to floats
    :param filename: name of data file
    :return: dataset (list of lists)
    '''
    with open(filename, 'r', encoding='UTF-8') as file:
        data = file.readlines()
    return_set = []
    for line in data:
        temp = []
        for val in line.split(','):
            try:
                temp.append(float(val))
            except ValueError:
                temp.append(val.replace('\n', ''))
        return_set.append(temp)
    return return_set


def unique(not_unique_list):
    '''
    :param not_unique_list:
    :return: unique list
    '''
    return list(dict.fromkeys(not_unique_list))


def str_column_to_int(dataset, column):
    '''
    Converts string column to an integer
    :param dataset: list of lists
    :param column: column number
    :return: modified dataset
    '''
    class_values = [row[column] for row in dataset]
    unique_class_values = unique(class_values)
    lookup = dict()
    for i, class_value in enumerate(unique_class_values):
        lookup[class_value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def split_dataset(dataset, percentage, num_of_classes, class_set_size):
    '''
    Assumption - each class has equal representation in dataset and dataset is sorted (by class)
    :param num_of_classes:
    :param class_set_size:
    :param dataset: list of lists
    :param percentage: size in % of train set (0 < float < 1)
    :return: train_set, test_set
    '''
    data = dataset.copy()
    size = round((len(data) * percentage)/num_of_classes)  # 3 classes
    # slicing
    train_set = []
    test_set = []
    for i in range(num_of_classes):
        train_set += data[class_set_size*i:(class_set_size*i)+size]
        test_set += data[(class_set_size*i)+size:(class_set_size*i)+class_set_size]
    return train_set, test_set


def cross_validation_split(dataset, n_folds):
    '''
    Splits dataset int k folds choosing random items
    :param dataset: list of lists
    :param n_folds: number of folds
    :return: split dataset into n folds (list of lists)
    '''
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


def accuracy_metric(actual, predicted):
    '''
    Calculates accuracy percentage
    :param actual: list of actual classes
    :param predicted: list of predicted classes
    :return: percentage of accuracy
    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    Evaluates an algorithm using a cross validation split
    :param dataset: list of lists
    :param algorithm: function
    :param n_folds: number of folds to cross validation
    :param args:
    :return: scores
    '''
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
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
        print(f'Predicted: {predicted}')
        actual = [row[-1] for row in fold]
        print(f'Actual:    {actual}')
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def evaluate_algorithm2(dataset, algorithm, n_folds, percentage, num_of_classes, class_set_size, sh):
    '''
    Evaluates an algorithm (without cross validation)
    :param dataset: list of lists
    :param algorithm: function
    :param n_folds: number of folds to cross validation
    :param percentage: percentage: size in % of train set (0 < float < 1)
    :param num_of_classes:
    :param class_set_size:
    :param sh: if shuffle (bool)
    :return: scores
    '''
    train_set, test_set = split_dataset(dataset, percentage, num_of_classes, class_set_size)
    if sh:
        shuffle(train_set)
    actual = [row[-1] for row in test_set]
    scores = list()
    predicted = algorithm(train_set, test_set)
    print(f'Predicted: {predicted}')
    print(f'Actual:    {actual}')
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    return scores


def separate_by_class(dataset):
    '''
    Separates data by class
    :param dataset: list of lists
    :return: dictionary of data grouped by class
    '''
    data_separated_by_class = dict()
    for data_row in dataset:
        class_value = data_row[-1]    # assumption: class name is the last value in each row of dataset
        if class_value not in data_separated_by_class:
            data_separated_by_class[class_value] = list()
        data_separated_by_class[class_value].append(data_row)
    return data_separated_by_class


def avg(numbers):
    '''
    Calculates average
    :param numbers: list of numbers
    :return: average
    '''
    return sum(numbers)/float(len(numbers))


def std_dev(numbers):
    '''
    Calculates standard deviation
    :param numbers: list of numbers
    :return: standard deviation
    '''
    average = avg(numbers)
    variance = sum([(x - average) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def get_dataset_stats(dataset):
    '''
    Calculates average, standard deviation and count for each column in dataset
    :param dataset: list of lists
    :return: stats (list of tuples)
    '''
    stats = [(avg(column), std_dev(column), len(column)) for column in zip(*dataset)]
    del(stats[-1])
    return stats


def get_dataset_stats_by_class(dataset):
    '''
    Splits dataset by class and perform summary
    :param dataset: list of lists
    :return: stats grouped by class
    '''
    separated = separate_by_class(dataset)
    stats = dict()
    for class_value, rows in separated.items():
        stats[class_value] = get_dataset_stats(rows)
    return stats


def calculate_probability(x, average, std_deviation):
    '''
    Calculates the Gaussian probability distribution function for given x
    :param x: number
    :param average:
    :param std_deviation: standard deviation
    :return: probability
    '''
    exponent = exp(-((x-average)**2 / (2 * std_deviation**2)))
    return (1 / (sqrt(2 * pi) * std_deviation)) * exponent


def calculate_probabilities_by_class(stats, row):
    '''
    Calculates the probabilities of predicting each class for a given row
    :param stats: stats by class
    :param row: dataset row
    :return: dictionary of probabilities with entry for each class
    '''
    total_rows = sum([stats[label][0][2] for label in stats])   # total number of training records
    probabilities = dict()
    for class_value, class_stats in stats.items():
        probabilities[class_value] = stats[class_value][0][2]/float(total_rows)     # P(class)
        for i in range(len(class_stats)):
            average, std_deviation, count = class_stats[i]
            probabilities[class_value] *= calculate_probability(row[i], average, std_deviation)     # P(class) * P(X|class)...
    return probabilities


def predict_class_for_row(stats, row):
    '''
    Predicts the class for a given row of dataset
    :param stats: stats by class
    :param row: dataset row
    :return: predicted class
    '''
    probabilities = calculate_probabilities_by_class(stats, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def naive_bayes(train_dataset, test_dataset):
    '''
    :param train_dataset:
    :param test_dataset:
    :return: predictions
    '''
    summarize = get_dataset_stats_by_class(train_dataset)
    predictions = list()
    for row in test_dataset:
        output = predict_class_for_row(summarize, row)
        predictions.append(output)
    return predictions


def print_graph(dataset, row_x, row_y, row_x_info, row_y_info):
    x_values = []
    y_values = []
    colors = []
    for data_row in dataset:
        x_values.append(data_row[row_x])
        y_values.append(data_row[row_y])
        if data_row[-1] == 0:
            colors.append('red')
        elif data_row[-1] == 1:
            colors.append('blue')
        else:
            colors.append('green')
    for x, y, col in zip(x_values, y_values, colors):
        plt.scatter(x, y, c=col, s=3, linewidth=0)

    red_p = mpatches.Patch(color='red', label='Iris Setosa')
    blue_p = mpatches.Patch(color='blue', label='Iris Versicolour')
    green_p = mpatches.Patch(color='green', label='Iris Virginica')
    plt.legend(handles=[red_p, blue_p, green_p], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel(row_x_info)
    plt.ylabel(row_y_info)
    plt.show()


def analyze_dataset():
    dataset = load_data('bezdekIris.data')
    str_column_to_int(dataset, 4)
    print_graph(dataset, 0, 1, 'sepal length', 'sepal width')
    print_graph(dataset, 0, 2, 'sepal length', 'petal length')
    print_graph(dataset, 0, 3, 'sepal length', 'petal width')
    print_graph(dataset, 1, 2, 'sepal width', 'petal length')
    print_graph(dataset, 1, 3, 'sepal width', 'petal width')
    print_graph(dataset, 2, 3, 'petal length ', 'petal width')

# constants
FOLDS_N = 5


def run():
    '''
    Runs program. Train set and test set chosen with cross validation.
    '''
    dataset = load_data('bezdekIris.data')
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    scores = evaluate_algorithm(dataset, naive_bayes, FOLDS_N)
    print(f'Scores: {scores}')
    print(f'Average accuracy: {round(sum(scores)/float(len(scores)), 2)}%')


PERCENTAGE = 0.3
NUM_OF_CLASSES = 3
CLASS_SET_SIZE = 50
SHUFFLE = True


def run2():
    '''
    Runs program. Checks proportions of train dataset and test dataset
    '''
    percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = load_data('bezdekIris.data')
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # evaluate algorithm
    scores = []
    for p in percentages:
        score = evaluate_algorithm2(dataset, naive_bayes, FOLDS_N, p, NUM_OF_CLASSES, CLASS_SET_SIZE, SHUFFLE)
        print(f'Score: {score}')
        scores.append(score)
    plt.plot(percentages, scores, 'r*')
    plt.xlabel('percentage number of train dataset')
    plt.ylabel('scores')
    plt.show()


if __name__ == '__main__':
    # run()
    # analyze_dataset()
    run2()
