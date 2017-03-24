"""
network.py
~~~~~~~~~~

Simple neural net implementation from Michael Nielsen's
Neural Networks and Deep Learning book. Here, I've retyped
all the code and commented each line for understanding.

"""

import random #standard lib

import numpy as np #np for linalg

class Network(object):

    def __init__(self, sizes):
        """Initialize the network"""
        # Store number of layers in the net
        self.num_layers = len(sizes)

        # Array to store the sizes of each layer
        self.sizes = sizes

        # Make a column vector of biases drawn from gaussian(0,1)
        # for each layer in the network except the input layer.
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # Let's see what this is doing - say sizes = [2, 5, 3]
        # Then the zip term returns (2,5),(5,3).
        # Then we take these pairs and use them to make, for ex
        # a 5 x 2 matrix. Each array in the outside array
        # represents a layer, and each row in the inside array
        # represents a neuron in that layer, each column in
        # that row represents the weight assigned to the
        # connection coming from that neuron in the previous
        # layer. We have a fully connected graph here.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Process the output of the network for an input a"""
        # For each layer in the network, get a pair consisting
        # of the column vector of biases for that layer, and
        # an array of weights having the same number of rows.
        for b, w in zip(self.biases, self.weights):
            # Take the matrix product of the input and the weights
            # and add the biases.
            a = sigmoid(np.dot(w, a) + b)

            #Here's an example
            # 1 2     2       6
            # 3 4  X  2   =  14
            # 5 6            22
            #
            #
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # One epoch involves descending w.r.t every point in the
        # training set.
        for j in xrange(epochs):
            # Shuffle the training data
            random.shuffle(training_data)
            # Split the data up into mini_batches.
            # Note - mini batch size should divide number of
            # data points here for this to include all data
            # in each epoch.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # We update the weights and biases of the net
                # using this mini batch. Here, eta is the
                # learning rate.
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # Evaluate the net on the test_data and output
                # performance for each epoch. This will slow
                # performance considerably.
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                # Print out progress
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        # Container for the gradient of the biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Container for the gradient of the weights
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Iterate through the points in the mini batch
        # Remember:
        # nabla_C = [nabla_b, nabla_w] ~ sum_{i in train_data} C_i
        # Where C_i is cost on a particular training example.
        for x, y in mini_batch:
            # Obtain the gradient of the cost function w.r.t
            # this specific data point.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Update the gradient of the biases
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # Update the gradient of the weights
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Make the gradient descent step on both the weights and
        # the biases for this mini batch. That is, update the actual
        # weights and biases of the network according to the
        # "law of motion" for "rolling downhill".
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    # TODO: Understand backpropagation algorithm.
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # Our network makes its classification based on which
        # output neuron has the highest value - this line generates
        # a list of tuples (predicted class, actual class).
        test_results = [(np.argmax(self.feedforward(x)),y)
                        for (x,y) in test_data]
        # Return the number of correct results.
        return sum(int(x == y) for (x, y) in test_results)

    # TODO comment
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    # Output 1/(1+e^-z)
    return 1.0/(1.0+np.exp(-z))

# TODO comment
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
