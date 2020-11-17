'''
MI Assignment 3
TEAM: Arsh Goyal PES1201800254 ; Sujith K PES1201802029
The Neural Network consists of 5 layers -- 1 input, 3 hidden and 1 output layer
The input dataset has 7 columns (Columns Education and Delivery phase were dropped)
Hence in the neural network,
Input layer has 7 neurons and uses the RelU activation function
Hidden Layer 1 has 12 neurons and uses the RelU activation function
Hidden Layer 2 has 24 neurons and uses the RelU activation function
Hidden Layer 3 has 8 neurons and uses the sigmoid activation function
Output Layer has 1 neuron which is the predicted value of the variable.
The loss function is Mean Squared Error

The dimensions of the weight matrices are:
    For Hidden Layer 1 : 7:12
    For Hidden Layer 2 : 12:24
    For Hidden Layer 3 : 24:8
    For Output Layer : 8:1

The bias matrices are:
    For Hidden Layer 1 : 12:1
    For Hidden Layer 2 : 24:1
    For Hidden Layer 3 : 8:1
    For Output Layer : 1:1
    
Steps taken in data pre-processing
    1. We replaced the NULL values with median for numerical data and mode for categorical data.
    2. We dropped two columns 'Education' and 'Delivery phase'.
    3. We removed outliners from BP column.
    4. We repopulated the dataset with rows containing result as 0 to avoid training bias.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

###################################################################################################
#Defining Activation Functions
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

###################################################################################################
#Defining Loss Functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

###################################################################################################
# This is the abstract base class laye. All other layers inherit from this.
# It handles properties such as an input, an output, and both a forwardpropogation and backward propogartion method.
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Given a input X, computes the output Y
    def forward_propagation(self, input):
        raise NotImplementedError

    # Updates the parameters if any and computes dE/dX for a given dE/dY
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
        
###################################################################################################        
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # The activation function taked the data as input and returns the activated output.
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # For a given output_error=dE/dY, the input_error=dE/dX is returned 
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
###################################################################################################  
class FCLayer(Layer):

    # Here, for any layer, 
    # input_size = number of input neurons to that layer
    # output_size = number of output neurons to that layer
    # The weights and bias for a layer are randomly initialized.
    # Shape of Weight matrix is Input_size:Output_size
    # Shape of bias matrix is 1:Output_size
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # Performs forward propogation for a given input:
    # Output = (Input * Weights) + Bias
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # Performs backword propogation for a given input:
    # Computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # The input_error=dE/dX is returned.
    # The parameters are also updates
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

###################################################################################################
class NN:
    
    # Initialize a empty list of layers and loss functions as None
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        
    # Function to add a layer to network
    def add(self, layer):
        self.layers.append(layer)

    # Function to set the loss function to use.
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
      
    # Function which actually trains the network by running a training loop.
    def train_network(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        # Training the Neural Network(loop)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Calculating the average error.
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
       
    # Calls the function train_network which performs the actual training
    def fit(self,X,y):
        self.train_network(X,y,epochs=40, learning_rate=0.1)
        
    # Function to pedict the output for the given dataset,
    def predict(self, X):
        # sample dimension first
        samples = len(X)
        result = []

        # Running the nueral network over all samples
        for i in range(samples):
            # Running forward propagation
            output = X[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    # Function to print the Confusion Matrix and other characteristics
    def CM(self,y_test,y_test_obs):
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
        
        acc=(tp+tn)/(tp+tn+fn+fp)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy : {acc}")
 
###################################################################################################
#FUnctio which reads the csv column and creates the dataframe.
#It also splits the datset in test and training data.
def create_df(filename):
    df = pd.read_csv('cleaned1.csv')
    df=(df-df.min())/(df.max()-df.min())
    #print(df.head())

    X = df.loc[:,'Community':'Residence'].values
    Y = df.loc[:,'Result'].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)
    
    x_train=x_train.reshape(len(x_train),1,7)
    y_train=y_train.reshape(len(y_train),1,1)
    
    return x_train, x_test, y_train, y_test 

###################################################################################################
#Main Function
if __name__ == "__main__":
    
    #Creating a dataframe from the cleaned csv file.
    #The function create_df also splits the dataframe into test and train datasets
    x_train, x_test, y_train, y_test = create_df('cleaned1.csv')  

    # Creating a object of the NN Class      
    net = NN()
    #Adding the input layer which 7 inputs, 12 outputs. Assigning it the ReLU activation function.
    net.add(FCLayer(7, 12))
    net.add(ActivationLayer(relu, relu_prime))
    
    #Adding the first hidden layer which 12 inputs, 24 outputs. Assigning it the ReLU activation function.
    net.add(FCLayer(12,24))
    net.add(ActivationLayer(relu, relu_prime))
    
    #Adding the second hidden layer which 24 inputs, 8 outputs. Assigning it the ReLU activation function.
    net.add(FCLayer(24,8))
    net.add(ActivationLayer(relu , relu_prime))
    
    #Adding the third hidden layer which 8 inputs, 1 output. Assigning it the sigmoid activation function.
    net.add(FCLayer(8,1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    
    #Assigning the loss function
    net.use(mse, mse_prime)
    
    #Fitting the neural network to the training data
    net.fit(x_train, y_train)
    
    #Predciting the values for the test data
    out = np.asarray(net.predict(x_test))
    
    #Checking the accuracy of results and printing the confusion matrix
    net.CM(y_test,out)
###################################################################################################





