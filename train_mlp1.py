import numpy as np
import mlp1 as mlp1
import utils as utils
import random

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def vector_normalization(vec):
    vector_sum = np.sum(vec)
    return np.divide(vec, vector_sum)


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    # find the number of id - how many different bigrams
    num_of_ids = len(utils.F2I)
    vec = np.zeros(num_of_ids)
    for bigram in features:
        if bigram in utils.F2I:
            m_id = utils.F2I[bigram]
            # update count
            vec[m_id] = 1 + vec[m_id]

    # we need to do normalization to the input layer in order to prevent
    # situation of to big numbers

    return vector_normalization(vec)


def CreatePredictionsFile(data, parameters):
    file_predictions = open("test.pred", 'w')
    # list of languages
    languages_list = utils.L2I.items()
    for tag, features in data:
        x = feats_to_vec(features)  # convert features to a vector.
        predicted_language = mlp1.predict(x, parameters)
        for language, text in languages_list:  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if predicted_language == text:
                tag = language
                break
        file_predictions.write(str(tag) + "\n")
        # close the file
    file_predictions.close()


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        vec_features = feats_to_vec(features)
        y_predict = mlp1.predict(vec_features, params)
        language_label = utils.L2I[label]
        if y_predict == language_label:
            good = good + 1
        else:
            bad = bad + 1
    return good / (good + bad)

def update_rule_params(grades,learning_rate,params):
    # Load parameters
    # w1 = params[0]
    # b1 = params[1]
    # w2 = params[2]
    # b2 = params[3]

    # Update parameters
    params[0] -= grades['dw1'] * learning_rate
    params[1] -= grades['db1'] * learning_rate
    params[2] -= grades['dw2'] * learning_rate
    params[3] -= grades['db2'] * learning_rate




def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            # convert features to a vector.
            x = feats_to_vec(features)
            #print (x.shape)
            language_label = utils.L2I[label]
            y = language_label  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            #print 'helllllll'
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            # update b matrix - rule update : b = b -n * gradientB

            update_rule_params(grads, learning_rate, params)
        #print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy

    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    # number of features - different bigrams - input layer dim
    in_dim = len(utils.F2I)
    # number of languages - output layer dim
    out_dim = len(utils.L2I)
    train_data = utils.TRAIN
    dev_data = utils.DEV
    num_iterations = 40
    learning_rate = 0.05
    hidden_dim = 20

    params = mlp1.create_classifier(in_dim,hidden_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    CreatePredictionsFile(train_data, trained_params)
