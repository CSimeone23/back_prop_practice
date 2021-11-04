from random import seed
from random import random
from random import randrange
from csv import reader
from math import exp

#################### Initialize a network ####################
def initialize_network(n_inputs, n_hidden, n_outputs):
    #Input Weights + Bias
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

#################### Forward Propagate ####################
#Neuron Activation
def activate(weights, inputs):
    # activation = sum(weight_i * input_i) + bias
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation 

#Neuron Transfer [SIGMOID ACTIVATION]
def transfer(activation):
    # output = 1 / (1+e^(-activation))
    return 1.0/(1.0+exp(-activation))

#Forward Propagation
# all outputs of one layer become the inputs for the next layer
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

#################### Back Propagate Error ####################

# Transfer Derivative
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    # Since we are using the sigmoid transfer function, the derivative
    # is calculated as seen below
    return output * (1.0 - output)

# Error Backpropagation
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

#################### Train Network ####################

# Update Weights
# weight = weight - learning_rate * error * input
# where weight is a given weight, learning_rate is a parameter
# that you must specify, error is the erro calculated by the
# back propagation procedure for the neuron and input
# is the input value that caused the error
# small learning rates with a large number of training iterations
# are preferred because it increases the likelihood of the network
# finding a good set of weights across all layers rather than the
# fastest set of weights that minimize error
# This is called Premature Convergence

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

#Train Network
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

#################### Predict ####################
# Make a prediction with the network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

#################### Wheat Seeds Dataset ####################

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Data normalization
def dataset_minmax(dataset):
    # Find the min and max values for each column
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    #Rescale dataset columns to the range 0-1
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Algorithm Evaluation
def cross_validation_split(dataset, n_folds):
    #Split a dataset into k folds
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # Validate using k-fold cross-validation
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
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def accuracy_metric(actual, predicted):
    # Calculate the accuracy of predictions
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Manage the application of the Backpropagation algorithm
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    # With Stochastic Gradient Descent
    n_inputs = len(train[0])-1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


#################### TESTING ####################
seed(1)
# Load seed dataset and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# Convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# Normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# Evaluate Algorithm
n_folds = 5
l_rate = 0.2
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
