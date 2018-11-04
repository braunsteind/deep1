import numpy as np

STUDENT = {'name': 'Daniel Braunstein',
           'ID': '312510167'}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    x = np.divide(e_x, np.sum(e_x))
    return x


def classifier_output(x, params):
    # YOUR CODE HERE.
    activation_i = x
    num_of_parameters = len(params)
    # loop with jumping 2, to get each time currents w and b
    for i in range(0, num_of_parameters, 2):
        W = params[i]
        b = params[i + 1]
        output = np.dot(activation_i, W) + b
        activation_i = np.tanh(output)
    probs = softmax(output)
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
    # probs = classifier_output(x, params)  # pred vec
    # h_s, z_s = fp(x, params)
    # loss = -np.log(probs[y])
    # gradients = []
    # y_one_hot = np.zeros(len(probs))
    # y_one_hot[y] = 1
    # grad_so_far = -(y_one_hot - probs)
    #
    # # grad of wn
    # gradients.append(np.outer(h_s.pop(), grad_so_far))
    #
    # # grad of bn
    # gradients.append(np.copy(grad_so_far))
    # # compute grad of all params
    # for i, (w, b) in enumerate(zip(params[-2::-2], params[-1::-2])):
    #     if (len(z_s) != 0):
    #         z_i = z_s.pop()
    #         w_i_plus_one = w
    #         if (len(h_s) != 0):
    #             h_i_minus_one = h_s.pop()
    #             # calcelate gradients
    #             dz_dh = w_i_plus_one
    #             dh_dz = 1 - np.square(np.tanh(z_i))
    #             dz_dw = h_i_minus_one
    #             grad_so_far = np.dot(grad_so_far, np.transpose(dz_dh)) * dh_dz
    #
    #             # grad of w
    #             gradients.append(np.outer(dz_dw, grad_so_far))
    #             # grad of b
    #             gradients.append(np.copy(grad_so_far))
    # rev_grad = []
    # for w, b in zip(gradients[0::2], gradients[1::2]):
    #     rev_grad.append(b)
    #     rev_grad.append(w)
    #
    # return loss, list(reversed(rev_grad))

    h = [x]
    for W_i, b_i in zip(params[0:-2:2], params[1:-1:2]):
        h.append(np.tanh(np.dot(h[-1], W_i) + b_i))
    y_hat = softmax(np.dot(h[-1], params[-2]) + params[-1])
    y_real = np.zeros(y_hat.shape)
    y_real[y] = 1

    loss = -np.log(y_hat[y])

    grads = []
    ### gradient of loss by y_hat
    g_until_now = -(y_real - y_hat)

    for i, (W_i, b_i) in enumerate(zip(params[-2::-2], params[-1::-2])):
        g_b_i = np.copy(g_until_now)
        g_w_i = np.outer(h[-i - 1], g_until_now)
        grads.append(g_b_i)
        grads.append(g_w_i)
        g_until_now = np.dot(W_i, g_until_now) * np.square(h[-i - 1])

    grads = list(reversed(grads))
    return loss, grads


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
    # prevent from repeating calculate
    sqrt_six = np.sqrt(6)
    parameters_list = []
    # dims_list[1:] is the same as dims_list just without first argument
    for x, y in zip(dims_list, dims_list[1:]):
        epsilon = sqrt_six / (np.sqrt(x + y))
        w = np.random.uniform(-epsilon, epsilon, [x, y])
        parameters_list.append(w)
        epsilon = sqrt_six / (np.sqrt(y))
        b = np.random.uniform(-epsilon, epsilon, y)
        parameters_list.append(b)
    return parameters_list
