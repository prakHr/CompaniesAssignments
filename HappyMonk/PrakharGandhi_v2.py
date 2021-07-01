from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
from pprint import pprint
import scipy
from scipy.special import logsumexp

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

#Softmax Activation
# correct solution:
def softmax_transfer(x):
    """Compute softmax values for each sets of scores in x."""
    #x[~np.isnan(x)] = x[~np.isnan(x)] > 709
    #out_vec=x
    #if np.isnan(np.sum(out_vec)):
    #    out_vec = out_vec[~np.isnan(out_vec)] # just remove nan elements from vector
    #out_vec[out_vec > 709] = 709
    #x=out_vec
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum() # only difference
    out_vec=x
    out_vec = np.exp(out_vec - logsumexp(out_vec))
    return out_vec
def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    import math
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*math.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))*math.sqrt(2./layers_dims[l-1])
        ### END CODE HERE ###
        
    return parameters

#parameters = initialize_parameters_he([1, 2, 2, 1])
#pprint(parameters)

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters

# Load a CSV file
def loadCsv(filename):
        trainSet = []
        
        lines = csv.reader(open(filename, 'r'))
        dataset = list(lines)
        for i in range(len(dataset)):
                for j in range(4):
                        #print("DATA {}".format(dataset[i]))
                        dataset[i][j] = float(dataset[i][j])
                trainSet.append(dataset[i])
        return trainSet

def minmax(dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
 
# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
        for row in dataset:
                for i in range(len(row)-1):
                        row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Convert string column to float
def column_to_float(dataset, column):
        for row in dataset:
                try:
                        row[column] = float(row[column])
                except ValueError:
                        print("Error with row",column,":",row[column])
                        pass
 
# Convert string column to integer
def column_to_int(dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i
        for row in dataset:
                row[column] = lookup[row[column]]
        return lookup
 
# Find the min and max values for each column

 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
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
 
# Calculate accuracy percentage
def accuracy_met(actual, predicted):
        correct = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        correct += 1
        return correct / float(len(actual)) * 100.0

def compute_loss(a3, Y):
    
    """
    Implement the loss function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    loss - value of the loss function
    """
    
    #m = Y.shape[1]
    m=len(Y)
    
    epsilon = 1e-5 
    logprobs = np.multiply(-np.log(a3+epsilon),Y) + np.multiply(-np.log(1 - a3+epsilon), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    return loss

def g(x,k0,k1,k2):
    """
    User defined activation function
    """
    ans=k0+k1*x
    return ans

def forward_propagation(X, parameters,k0,k1,k2):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
        
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X.T) + b1
    #print("z1",z1)
    a1 = g(z1,k0,k1,k2)
    #print("a1",a1)
    z2 = np.dot(W2, a1) + b2
    #print("z2",z2)
    a2 = g(z2,k0,k1,k2)
    #print("a2",a2)
    z3 = np.dot(W3, a2) + b3
    #print("z3",z3)
    a3 = softmax_transfer(z3)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    
    return a3, cache

def backward_propagation_error(X, Y, cache, k1):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    #m=len(X)
    #print("m",m)
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, k1)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    #aa=np.multiply(Z2,dA2)
    #bb=np.array(np.multiply(np.array(Z2*Z2,dtype=np.float64),np.array(dA2,dtype=np.float64)),dtype=np.float64)
    aa=Z2*dA2
    bb=Z2*aa
    #print("aa",aa)
    #print("bb",bb)
    dK2 =np.array([1./(dA2.shape[0]*dA2.shape[1]) * np.sum(dA2),
          1./(aa.shape[0]*aa.shape[1]) * np.sum(aa),
          1./(bb.shape[0]*bb.shape[1]) * np.sum(bb)])
    dA1 = np.dot(W2.T, dZ2)
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1,k1)
    dW1 = 1./m * np.dot(dZ1, X)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1,"dK2":dK2}
    
    return gradients    

def update_params(W1, b1, W2, b2, W3, b3, K, dW1, db1, dW2, db2, dW3, db3, dA1,dA2,Z1,Z2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3

    #aa=np.multiply(Z1,dA1)
    #bb=np.array(np.multiply(np.array(Z1*Z1,dtype=np.float64),np.array(dA1,dtype=np.float64)),dtype=np.float64)
    aa=Z1*dA1
    bb=Z1*aa
    
    dK1 =np.array([1./(np.size(dA1)) * np.sum(dA1),
          1./(np.size(aa)) * np.sum(aa),
          1./(np.size(bb)) * np.sum(bb)])
    
    #cc=np.multiply(Z2,dA2)
    #dd=np.array(np.multiply(np.array(Z2*Z2,dtype=np.float64),np.array(dA2,dtype=np.float64)),dtype=np.float64)
    cc=Z2*dA2
    dd=Z2*cc
    
    dK2 =np.array([1./(np.size(dA2)) * np.sum(dA2),
          1./(np.size(cc)) * np.sum(cc),
          1./(np.size(dd)) * np.sum(dd)])
    dK=dK1+dK2
    #dK[0] = dK1[0] + dK2[0]
    #dK[1] = dK1[1] + dK2[1]
    #dK[2] = dK1[2] + dK2[2]
    
    K[0] = K[0]-alpha * dK[0]
    K[1] = K[1]-alpha * dK[1]
    K[2] = K[2]-alpha * dK[2]
    return W1, b1, W2, b2, W3, b3, K

def cross_entropy(x, y):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError('Negative values exist.')

    # Force to proper probability mass function.
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    x /= np.sum(x)
    y /= np.sum(y)

    # Ignore zero 'y' elements.
    mask = y > 0
    x = x[mask]
    y = y[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = r'C:\Users\Prakhar gandhi\Desktop\ProjectEuler_MengGenTsai\data.csv.txt'
dataset = loadCsv(filename)
for i in range(len(dataset[0])-1):
        column_to_float(dataset, i)
# convert class column to integers
column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = minmax(dataset)
normalize(dataset, minmax)

n_folds = 2
l_rate = 0.1
learning_rate=0.000000000000000000001
n_epoch = 10
n_hidden_1 = 4
n_hidden_2 = 4
print_cost=True
num_iterations=150000
#algorithm is backpropagation    
folds = cross_validation_split(dataset, n_folds)
#print("folds\n",folds)
scores = list()
for fold in folds:
    costs=[]
    k0=np.random.uniform(-2, 2, 1)
    k1=np.random.uniform(-2, 2, 1)
    k2=np.random.uniform(-2, 2, 1)
    K=np.array([k0,k1,k2])
    #print("Test Fold {} \n \n".format(fold))
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = list()
    for row in fold:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
    
        #a3, cache = forward_propagation(X, parameters)
        #print(a3)
    
    n_inputs = len(train_set[0]) - 1
    n_outputs = len(set([row[-1] for row in train_set]))
    parameters = initialize_parameters_he([n_inputs, n_hidden_1,n_hidden_2,n_outputs])
    #pprint(parameters)
    #print("-"*100)
    # W1,W2,W3,b1,b2,b3
    #parameters=network
    
    #pprint(network)
    #pprint(train_set)
    X,Y=[],[]
    for i in range(len(train_set)):
        my_list=train_set[i]
        X.append(my_list[:-1])
        Y.append(my_list[-1])
    X=np.array(X)
    Y=np.array(Y)
    #pprint(Y)

    #cache=(z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    for num in range(0,num_iterations):
        a3, cache = forward_propagation(X, parameters,k0,k1,k2)
        #print("a3",a3.shape)
        #print("cache",cache)
        #print(parameters)
        cost = compute_loss(a3,Y)
        #cost=cross_entropy(X, Y)

        #print("cost",cost)
        grads = backward_propagation_error(X, Y, cache,k1)
        #dA1, dA2,dK2,dW1,dW2,dW3,dZ1,dZ2,dZ3,db1,db2,db3
        #print("*"*100)
        #print("grads")
        #pprint(grads)
        #print("-"*100)
        #parameters = update_parameters(parameters, grads, learning_rate)
        
        W1, b1, W2, b2, W3, b3, K=update_params(np.array(parameters["W1"]), np.array(parameters["b1"]), np.array(parameters["W2"]), np.array(parameters["b2"]), np.array(parameters["W3"]), np.array(parameters["b3"]),
                                                    K, np.array(grads["dW1"]), np.array(grads["db1"]), np.array(grads["dW2"]),
                                                    np.array(grads["db2"]), np.array(grads["dW3"]), np.array(grads["db3"]),
                                                    np.array(grads["dA1"]),np.array(grads["dA2"]),Z1=np.array(cache[0]),Z2=np.array(cache[4]), alpha=learning_rate)
        #print("W1, b1, W2, b2, W3, b3",W1, b1, W2, b2, W3, b3)
        #print(type(W1), type(b1), type(W2), type(b2), type(W3), type(b3))
        #print("K",K)
        parameters={}
        parameters["W1"]=W1
        parameters["b1"]=b1
        parameters["W2"]=W2
        parameters["b2"]=b2
        parameters["W3"]=W3
        parameters["b3"]=b3
        #print("parameters")
        #pprint(parameters)
        #print("K")
        #pprint(K)
        k0,k1,k2=K[0],K[1],K[2]
        K=np.array([k0,k1,k2])
        #print("k0,k1,k2=>",k0,k1,k2)
        #print(W1, b1, W2, b2, W3, b3, K)
        if print_cost and num%1000==0:
            #print("parameters")
            #pprint(parameters)
            print("Cost after iteration {}".format(num))
            print(f'{cost:.20f}')
            costs.append(cost)
    print("costs",costs)
    print("Completed fold here")
