import mnist_loader
import network
import numpy as np
import cPickle as pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
weights, biases = net.SGD(training_data, 1, 1, 0.5, test_data=test_data)
pickle.dump(weights, open( "weights.p", "wb" ))
pickle.dump(biases, open( "biases.p", "wb" ))
