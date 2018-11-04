import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    activation_i = x
    num_of_parameters = len(params)
    #loop with jumping 2, to get each time currents w and b
    for index in range(0,num_of_parameters,2):


    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    return ...

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    params = init_parameters(dims)
    return params

def init_parameters(dims_list):
    #prevent from repeating calculate
    sqrt_six = np.sqrt(6)
    parameters_list=[]
    #dims_list[1:] is the same as dims_list just without first argument
    for x,y in zip(dims_list,dims_list[1:]):
        epsilon = sqrt_six/(np.sqrt(x+y))
        w=np.random.uniform(-epsilon,epsilon,[x,y])
        parameters_list.append(w)
        epsilon = sqrt_six/(np.sqrt(y))
        b=np.random.uniform(-epsilon,epsilon,y)
        parameters_list.append(b)
    return parameters_list



