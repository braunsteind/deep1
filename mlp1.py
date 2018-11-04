import loglinear as ll
import numpy as np

# check


STUDENT = {'name': 'Daniel Braunstein',
           'ID': '312510167'}


def classifier_output(x, params):
    # YOUR CODE HERE.
    feed_forward_model = forward_propagation_action(x, params)
    probs = feed_forward_model['A2']
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_deriv(y, y_hat):
    return y_hat - y


def tanh_deriv_function(x):
    return 1 - np.power(x, 2)


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
    # first load the model parameters
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]

    forward_model = forward_propagation_action(x, params)
    A0 = forward_model['A0']
    A1 = forward_model['A1']
    A2 = forward_model['A2']

    num_class = len(A2)

    y_vector = np.zeros(num_class)
    # print('y_vector:', y_vector.shape)
    # create a vector represent y
    y_vector[y] = 1

    db2 = loss_deriv(y_vector, A2)
    # print('db2:', db2.shape)
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

    dw2 = np.outer(db2, A1)
    # print('dw2:', dw2.shape)

    db1 = np.dot(db2.T, w2) * tanh_deriv_function(A1)
    # print('db1:', db1.shape)

    dw1 = np.outer(db1, A0)
    # print('A0:', A0.shape)
    # print('dw1:', dw1.shape)

    loss = -np.log(A2[y])
    # print('A2:', A2.shape)

    my_grades = {'dw2': dw2,
                 'db2': db2,
                 'dw1': dw1,
                 'db1': db1}

    return loss, my_grades


def init_uniform_parameter(epsilon, first_dim, second_dim):
    if second_dim == 1:
        return np.random.uniform(-epsilon, epsilon, first_dim)
    else:
        return np.random.uniform(-epsilon, epsilon, [first_dim, second_dim])


def init_parameters(in_dim, hid_dim, result_dim):
    # prevent from repeating calculate
    sqrt_six = np.sqrt(6)
    epsilon = sqrt_six / (np.sqrt(in_dim + hid_dim))

    # w1 size 30x600 , x: 600x1 -> wx: 30x1
    w1 = init_uniform_parameter(epsilon, hid_dim, in_dim)
    print('w1:', w1.shape)

    epsilon = sqrt_six / (np.sqrt(hid_dim))

    # b1 size: 30x1
    b1 = init_uniform_parameter(epsilon, hid_dim, 1)
    print('b1:', b1.shape)

    # w2: 10x30 A1: 30x1 -> w2xA1 : 10x1
    epsilon = sqrt_six / (np.sqrt(hid_dim + result_dim))
    w2 = init_uniform_parameter(epsilon, result_dim, hid_dim)
    print('w2:', w2.shape)

    epsilon = sqrt_six / (np.sqrt(result_dim))
    # b2: 10x1
    b2 = init_uniform_parameter(epsilon, result_dim, 1)
    print('b2:', b2.shape)

    return [w1, b1, w2, b2]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.


    """
    # A good boot method is to initialize in random values between Epsilon and Epsilon A good boot method is to
    # initialize in random values between Epsilon and Epsilon And initializes values in the middle range of the function rather than in the areas of the cytometry

    params = init_parameters(in_dim, hid_dim, out_dim)
    return params


def than(z):
    return np.tanh(z)


def forward_propagation_action(x, params):
    # first load the model parameters
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]

    # print('w1dotx:', np.dot(w1,x).shape)
    # print('x:', x.shape)
    # compute Z1: input layer matrix dot w1 wheight matrix plus our bias
    z1 = np.dot(w1, x) + b1
    # print('z1:', z1.shape)

    # put it throgh our activition function
    A1 = than(z1)
    # print('A1:', A1.shape)

    # compute Z2:
    z2 = np.dot(w2, A1) + b2
    # print('z2:', z2.shape)
    # now, we'll use the softmax as our activition function
    A2 = ll.softmax(z2)
    # print('A2:', A2.shape)
    # save all results as a model
    result_model = {'A0': x,
                    'z1': z1,
                    'A1': A1,
                    'z2': z2,
                    'A2': A2}

    return result_model
