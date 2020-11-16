'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import pandas as pd
from random import seed
from random import random
from math import exp
import sklearn
from sklearn.model_selection import train_test_split

class NN:
    ''' X and Y are dataframes '''

    def __init__(self):
        self.network=[]

    # Initialize a network
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        self.network = list()
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)
        return self.network

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(self, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                # print(outputs)
                expected = [0 for i in range(n_outputs)]
                expected[int(row[-1])] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            # if epoch%5==0:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    def test_predict(self, network, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))



    def fit(self, X, Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        seed(1)

        network=self.initialize_network(9,2,2)
        train_dataset = X
        train_dataset['Result'] = Y.copy()
        dataset = train_dataset.values.tolist()
        self.train_network(dataset, 0.3, 50, 2)


    def predict(self, X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values

        yhat is a list of the predicted value for df X
        """
        yhat=[]
        dataset=X.values.tolist()
        for row in dataset:
            prediction=self.test_predict(self.network,row)
            yhat.append(prediction)


        return yhat



    def CM(self, y_test, y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if (y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(y_test)):
            if (y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp + 1
            if (y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn + 1
            if (y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp + 1
            if (y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn + 1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


net=NN()

df=pd.read_csv('temp_cleaned.csv')

X=df[df.columns[:-1]]
y=df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

net.fit(X_train,y_train)
y_test_obs=net.predict(X_test)

net.CM(list(y_test), y_test_obs)

