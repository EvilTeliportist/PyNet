import numpy as np
import os

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, inputArray, answers, hidden, num_hidden_layers,  learn):
        self.input = inputArray
        self.answers = answers
        self.learning_rate = learn

        #necissary
        self.output = np.zeros(self.answers.shape)
        self.learn = learn
        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        self.NumOutputs = len(self.answers[0])
        self.training_counter = 0
        self.all_error = []

        #settings
        self.print_while_training = False
        self.percet_error = 0
        self.prompt_write = False

        self.make_weights()

    def make_weights(self):

        #initialize weights array
        self.weights = []

        #add a random array for the input --> hidden layer weights
        self.weights.append(np.random.rand(len(self.input[0]), self.hidden))

        #add all the weights in between the hidden layers
        for i in range(self.num_hidden_layers-1):
            self.weights.append(np.random.rand(self.hidden, self.hidden))

        #add weights from hidden layer --> output
        self.weights.append(np.random.rand(self.hidden, self.NumOutputs))

        #make random bias array including output bias
        self.bias = np.random.rand(self.num_hidden_layers+1)

    def feedforward(self, data):
        #initalize blank array to hold all node values
        self.values = []

        #add input to the values array
        self.values.append(data)

        #calculate values for each layer of nodes
        for i in range(self.num_hidden_layers +1):
            self.values.append(sigmoid(np.dot(self.values[-1], self.weights[i])+self.bias[i]))

        #set output to values of last layer
        self.output = self.values[-1]
        self.all_error.append((sum(self.output - self.answers)/len(self.output))**2)

        #toggle printing while training
        if self.print_while_training and self.training_counter % 100 == 0:
            print((sum(self.output-self.answers)/len(self.output))**2)
            self.training_counter += 1
        else:
            self.training_counter += 1

    def backpropagate(self):

        #initalize the error in the last layer
        error_in_layer = 2*(self.answers - self.output)*sigmoid_derivative(self.values[-1])

        #initialize array to hold values "delta weights", or change in weights
        dw = []

        #loop through each layer and calculate the error in that layer based on the layer after it, hence "back"-propagation
        for i in range(len(self.values)-1):
            delta = np.dot(self.values[-(i+2)].T, error_in_layer)*self.learning_rate
            dw.append(delta)
            error_in_layer = np.dot(error_in_layer, self.weights[-(i+1)].T) * sigmoid_derivative(self.values[-(i+2)])

        #change weights of each layer by dw (calculated for each layer)
        for i in range(len(dw)):
            self.weights[i] += dw[-(i+1)]

    def train(self, x):

        #train function that feedsforward and backpropegates in one function call
        for i in range(x):
            self.feedforward(self.input)
            self.backpropagate()

        #calculate percent error and print it
        self.percent_error = sum(self.answers - self.output)/len(self.answers)
        print('Average Percent Error In Last Set Of Output Data: ' + str(100*round(self.percent_error[0], 3))+'%')

        #setting to let user write weights to .txt
        if self.prompt_write:
            x = input('Write? (y/n)')
            if x == 'y':
                y = input('Input Filename')
                self.write_weights(y)

    def test(self, data):
        #initalize blank array to hold all node values
        self.values = []

        #add input to the values array
        self.values.append(data)

        #calculate values for each layer of nodes
        for i in range(self.num_hidden_layers +1):
            self.values.append(sigmoid(np.dot(self.values[-1], self.weights[i])+self.bias[i]))

        #set output to values of last layer
        self.output = self.values[-1]

        return self.output

    def write_weights(self, file_name):
        f = open(file_name, 'a')
        s = str(self.weights)
        f.write(s)

    def set_weights(self,  x):
        self.weights = x

    def print_weights(self):
        for i in range(len(self.weights)):
            print('Weights from layer ' + str(i) + ' to layer ' + str(i+1) + ':')
            print(self.weights[i])
