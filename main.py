import pynet
import os, numpy
import yfinance as yf
import matplotlib.pyplot as plt


#get initial data
ticker = yf.Ticker('MSFt')
raw = [i[1] for i in ticker.history(period='max').to_numpy()]


# split raw into sets of 5 days to train the network with
c = 0
split_raw = []
temp = []
for i in raw:
    if c < 5:
        temp.append(i)
        c += 1
    else:
        split_raw.append(temp)
        temp = []
        c = 0


# select only the second half of the data to get rid of very low prices that have large
# percentage based variation (ex $.06 and $.07 vs $22.50 and $22.51)
split_raw = split_raw[len(split_raw)//2:]


# normalize each set of data, preparing for input into the neural network
split_raw = [(set/numpy.linalg.norm(numpy.asarray(set))).tolist() for set in split_raw]


# create an array of expected outputs to train the network with
# expected outputs of 1 represent a good buy, while outputs of
# 0 represent a bad buy
exp_out = []
for set in split_raw:
    if set[-1] > set[-2]:
        exp_out.append([1])
    else:
        exp_out.append([0])

test_out = numpy.array(exp_out[len(exp_out)//2:])
exp_out = numpy.array(exp_out[:len(exp_out)//2])


# remove the last datapoint from each set because they are acting
# as the "answers" and we should not train the network on data that
# we wouldn't necissarily have access to during implementation
split_raw = [set[:-1] for set in split_raw]
split_raw = numpy.array(split_raw)

# creates a testing and training dataset by chopping the data in half
training = split_raw[:len(split_raw)//2]
testing = split_raw[len(split_raw)//2:]


# initializing neural network class
network = pynet.NeuralNetwork(training, exp_out, 2, 2, .0001)
network.print_while_training = True;

# train
network.train(1000)

# show a plot of error over time
plt.plot(network.all_error)
plt.show()

# test by inputting the test set into
test_output = network.test(testing)
average_test_error = sum(test_output - test_out)/len(test_out)
print("Average Test Error: " + str(round(100*average_test_error[0], 4)) + "%")
