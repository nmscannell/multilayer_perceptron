import scipy.io
digits = scipy.io.loadmat('digits.mat')
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

"""
Neural Networks Project 2: Using an architecture consisting of 784 input neurons,
25 hidden neurons, and 10 output neurons, classify images from the MNIST dataset. 
Each neuron i should return 1 when the image is a representation of i, and 0 otherwise.
Use back-propagation and logistic function (p 168) for activation; alpha can be 0.

Natasha Scannell
"""

train_x = digits['train']
train_x = train_x.astype('float32')/255
train_y = digits['trainlabels']
test_x = digits['test']
test_x = test_x.astype('float')/255
test_y = digits['testlabels']
layers = [784, 25, 10]
weights = []
bias = []
l_r = 0.1
epochs = 50
batch_size = 32
number_of_batches = int(len(train_y)/batch_size)


def train(x, d, learning_rate):
    # Given input samples and desired values, train network. Use batch training over epochs. Shuffle input for each epoch.
    indices = np.arange(len(d))
    errors_test = []
    errors_train = []
    for e in range(epochs):
        total_train_correct = 0
        shuffle(indices)
        for i in range(number_of_batches):
            start = i * batch_size
            end = min(start + batch_size, len(d))
            sum_dbh = 0
            sum_dwh = 0
            sum_dbo = 0
            sum_dwo = 0
            for j in range(start, end):
                y = forward(x[:, indices[j]])
                dbh, dwh, dbo, dwo = backprop(y, d[indices[j]])
                sum_dbh += dbh
                sum_dwh += dwh
                sum_dbo += dbo
                sum_dwo += dwo
                if d[indices[j]] == np.argmax(y[-1]):
                    total_train_correct += 1
            bias[2] += (learning_rate/batch_size) * sum_dbo
            weights[2] += (learning_rate/batch_size) * sum_dwo
            bias[1] += (learning_rate/batch_size) * sum_dbh
            weights[1] += (learning_rate/batch_size) * sum_dwh
        errors_test.append((len(test_y) - evaluate(test_x, test_y))/len(test_y))
        errors_train.append((len(d) - total_train_correct)/len(d))

        learning_rate = l_r * np.exp(-0.1*e)

    return errors_train, errors_test


def evaluate(x, d):
    # Test trained network using test samples and labels. Return accuracy rate.
    res = 0
    for i in range(len(d)):
        y = forward(x[:, i])
        out = y[-1]
        if d[i] == np.argmax(out):
            res += 1
    return res


def initialize_w():
    # Weights should be initialized with values from uniform distribution with mean 0. Each neuron in each layer needs enough weights for its input (number of neurons in prev layer).
    weights.append([])
    bias.append([])
    for i in range(1, len(layers)):
        num_prev = layers[i-1]
        num_current = layers[i]
        weights.append(np.random.normal(0, np.sqrt(2/(num_prev+num_current)), (num_current, num_prev)))
        bias.append(np.random.normal(0, np.sqrt(2/(num_prev+num_current)), (num_current, 1)))


def forward(x):
    # Evaluate activation function (sigmoid) at each neuron for weights*input at that layer. Return values.
    y = []
    y.append(x.reshape((layers[0], 1)))
    y.append(sigmoid(np.matmul(weights[1], y[0])) + bias[1])
    out = sigmoid(np.matmul(weights[2], y[1])) + bias[2]
    out_norm = np.zeros((10, 1))
    out_norm[np.argmax(out)] = 1
    y.append(out_norm)
    # y.append(np.round(sigmoid(np.matmul(weights[2], y[1])) + bias[2]))
    return y


def backprop(y, d):
    # Get error at last layer. Compute local gradients for each layer and return deltas for each bias/weight.
    d = to_column(d)
    o = y[2]
    error = d - o

    # v is input for sigmoid at output layer, output layer weights * output layer input
    v = np.matmul(weights[2], y[1])

    # local gradient for output layer is error * sig derivative(v)
    local_gradient_out = error * sigmoid_derivative(v)

    # bias gets adjusted with local_gradient * eta
    deltabo = local_gradient_out

    # delta is local_gradient * input for output layer
    deltawo = np.matmul(local_gradient_out, y[1].T)

    # v is input for sigmoid for hidden layer, hidden layer weights * hidden layer input
    v = np.matmul(weights[1], y[0])

    # local_gradient of the hidden layer is sig deriv(v) * sum(local_gradient_out * weights of the output layer)
    local_gradient_hidden = sigmoid_derivative(v) * np.matmul(local_gradient_out.T, weights[2]).T

    # delta is local_gradient
    deltabh = local_gradient_hidden

    # delta is local_gradient * input for hidden layer
    deltawh = np.matmul(local_gradient_hidden, y[0].T)

    return deltabh, deltawh, deltabo, deltawo


def sigmoid(z):
    # Activation function for each neuron. Return 1/[1 + exp(-v(n))]
    return 1/(1 + np.exp(-z))


def sigmoid_derivative(z):
    # Needed to calc local gradient for each neuron. Deriv = y(n)[1-y(n)].
    return sigmoid(z) * (1-sigmoid(z))


def to_column(d):
    c = np.zeros((10, 1))
    i = d
    c[i] = 1
    return c


initialize_w()
errors_train, errors_test = train(train_x, train_y, l_r)
num_correct = evaluate(train_x, train_y)
print("Accuracy on Train Data: ", num_correct/len(train_y))
num_correct = evaluate(test_x, test_y)
print("Accuracy on Test Data: ", num_correct/len(test_y))

plt.figure(figsize=(8, 8))
plt.plot(range(epochs), errors_train, label='Training Errors')
plt.plot(range(epochs), errors_test, label='Testing Errors')
plt.title('Training and Test Errors')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.ylim(0, 1)
plt.savefig('mlp_mini_batch_errors_lr_1.png')
plt.show()
