import loglinear as ll
import numpy as np


STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    feed_forward_model = forward_propagation_action(x,params)
    probs =  feed_forward_model['A2']
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    return ...

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = [W,b,U,b_tag]
    return params

def than(z):
    return np.tanh(z)

def forward_propagation_action(x,params):
    # first load the model parameters
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]

    # compute Z1: input layer matrix dot w1 wheight matrix plus our bias
    z1 = np.dot(w1,x)+b1

    # put it throgh our activition function
    A1 = than(z1)

    # compute Z2:
    z2 =np.dot(w2,A1)+b2

    # now, we'll use the softmax as our activition function
    A2 = ll.softmax(z2)

    # save all results as a model
    result_model = {'A0': x,
                    'z1': z1,
                    'A1': A1,
                    'z2': z2,
                    'A2': A2}


    return result_model



