import loglinear as log_linear
import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    activation_i = x
    num_of_parameters = len(params)
    #loop with jumping 2, to get each time currents w and b
    for index in range(0,num_of_parameters,2):
        matrix = params[index+1] + np.dot(activation_i,params[index])
        activation_i = np.tanh(matrix)
    probs = log_linear.softmax(matrix)

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
    grads_list= []
    An = classifier_output(x,params)
    #computing the loss
    m_loss = -np.log(An[y])
    list_z,list_A = forward_propagation_action(x,params)

    num_class = len(An)
    y_vector = np.zeros(num_class)
    # create a vector represent y
    y_vector[y] = 1
    dbn = loss_deriv(y_vector, An)
    cure_gradient = dbn

    An_minus_1 = list_A.pop()
    dwn = np.outer(An_minus_1,dbn)

    #update grads list:
    grads_list.append(dwn)
    grads_list.append(dbn)

    #now lets continue computing the rest of gradient until we get to dw1 and db1
    for index, (W,b) in enumerate(zip(params[-2::-2],params[-1::-2])):
        if(0 != len(list_z)):
            w_next = W
            cure_z_matrix = list_z.pop()
            if(0!=len(list_A)):
                A_prev = list_A.pop()
                cure_gradient = np.dot(cure_gradient,np.transpose(w_next))*tanh_deriv_function(cure_z_matrix)

                #update grads list:
                dw_cure = np.outer(A_prev,cure_gradient)
                db_cure = cure_gradient

                grads_list.append(dw_cure)
                grads_list.append(db_cure)
    reverse_gradients_params = reverse_grads_list(grads_list)
    return m_loss,reverse_gradients_params



def reverse_grads_list(grads):
    grads_list = []
    for W,b in zip(grads[0::2],grads[1::2]):
        grads_list.append(b)
        grads_list.append(W)
    return list(reversed(grads_list))

def loss_deriv(y,y_hat):
    return y_hat - y

def tanh_deriv_function(x):
    return 1 - np.square(np.tanh(x))

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

def forward_propagation_action(x, parameters):
    activation_cure_matrix = x
    num_of_parameters = len(parameters)
    matrix_after_comp_list = []
    activation_matrix_list = []
    activation_matrix_list.append(activation_cure_matrix)
    for index in range(0,num_of_parameters,2):
        matrix = parameters[1+index] + np.dot(activation_cure_matrix,parameters[index])
        matrix_after_comp_list.append(matrix)

        activation_cure_matrix = np.tanh(matrix)
        activation_matrix_list.append(activation_cure_matrix)
    #we can pop out the last ones because we get them from classifier_output
    matrix_after_comp_list.pop()
    activation_matrix_list.pop()
    return  matrix_after_comp_list,activation_matrix_list



