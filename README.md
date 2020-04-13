# PyNet
#### PyNet is a simple open source python package that allows for the implementation of neural networks in python. The main focus of PyNet is to give beginners an easy way to explore the possibilities of neural networks in code. It is recommended to have a basic understanding of how neural networks work, including terms like weights, layers, nodes, the sigmoid function, and activation. It is also reccommended to have experience in numpy as well.

#### For further understanding of how the network works in principle, check out these videos by a youtuber called 3Blue1Brown on the conceptualizations of neural networks: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

#### The code is written not for optimization, but for ease of reading and understanding. The primary focus of PyNet is not to be used in large scale applications or data analysis, but for beginners to play around with, recognize the potential of neural networks, and explore the code to deepen their knowledge of python

## Installation Instructions
#### Download the main.py file and move it into the folder which holds all of your python packages. You can now implement pynet like any other python package. 

### IMPORTANT: PyNet does not yet support bias in the layers yet.


## Example Code
### Models a simple XOR function

```
import neural_network as nn
import numpy as np

#establishes input array of all possible inputs to a binary dual-input network
x = np.array([[0,0], [0, 1], [1, 0], [1, 1]])

#establish array of the answers (correct output) for each pair of inputs
#must be a 2D array because the network supports multiple outputs
y = np.array([[0], [1], [1], [0])

#initializes a new neural network class with inputs:
# x = input array
# y = answers
# 3 nodes in each hidden layer
# 2 hidden layers
# learning rate of .1
n = nn.NeuralNetwork(x, y, 3, 2, .1)

#trains the network 10,000 times
n.train(10000)

```

**Neural Network Class**
- **Initialization Inputs**
  - **inputArray**
        - Array of all the input data. It is important to feed the neural network all of the input at once, or else it will train to replicate only the first thing you input. Array must be a numpy array.
  - **answers**
        - Array of the answers to each input, in order. See example code of XOR function. Must be a numpy array.
  - **hidden**
        - Number of nodes in each hidden layer
  - **num_hidden_layers**
        - Number of hidden layers in the network
  - **learn**
        - Learning Rate of the network
- **Functions**
  - **make_weights()**
        - Makes an array of random weights for the network.
  - **feedforward()**
        - Feed forward all of the input into the network, updating the network’s output values
        - Uses a sigmoid activation function on each layer, including the output
  - **backpropagate()**
        - Performs a single iteration of back propagation on the network, updating the weights to reduce the cost of the network.
  - **train(x)**
        - Runs feedforward() and backpropagate() x times
        - Prints the percent error in the last registered output
  - **write_weights(file_name)**
        - Creates a file ‘file_name’ and writes the current weights in that file.
        - Useful for saving trained networks 
  - **set_weights(x)**
        - Sets all weights for the network to x
        - X must be a numpy array of the proper size for your network
  - **print_weights()**
        - Prints all weights

- **Variables**
  - **print_while_training**
        - Boolean value, default of false, when true it will print the output array after every call of feedforward()
  - **percent_error**
        - The percent error after each call of train()
        - What train() prints at the end of each call
  - **prompt_write**
        - Boolean of whether to prompt user about writing the weights to a file after running train()
        - Only works in a console style application that accepts input
        - Default of false
  - **values**
        - array of the values each node holds AFTER ACTIVATION for each iteration of feedforward()
        - Holds arrays representing each layer (see diagram)
  - **weights**
        - numpy array of all weights in the network
        - Hold arrays of weights in each node in each layer
        - weights[layer][node] returns the weights of the node in specified layer
        - IMPORT TO NOTE THAT THERE ARE LESS WEIGHTS LAYERS THAN VALUES LAYERS (see diagram)

- **Package Functions**
  - **sigmoid(x)**
        - Returns sigmoid of x
  - **sigmoid_derivative(x)**
        - Returns the derivative of sigmoid(x), which is just (x*(1-x))
