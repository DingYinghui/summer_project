# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
# ONE problem: THE INPUT DATA CONSTRUCT:
#   https://www.tensorflow.org/api_guides/python/reading_data

import tensorflow as tf
import pandas as pd
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data


# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
file_name = "./data/raw_refeatured_data.csv"

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


n_classes = 1
batch_size = 100

x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32)


def correct_csv():
    df = pd.read_csv(file_name, header=0, index_col=0)

    local_dict = { "medium": 1}
    for i in range(len(df.iloc[:, -1])):
        df.iloc[i, -1] = local_dict[df.iloc[i, -1]]

    df.to_csv(file_name)


def read_from_csv():
    df = pd.read_csv(file_name, header=0, index_col=0)

    train_X = np.array(df.iloc[: 5000, : -1].values, dtype=np.float)
    train_Y = np.array(df.iloc[: 5000, -1].values, dtype=np.float32)
    train_Y = train_Y.reshape(train_Y.shape[0], 1)

    test_X = np.array(df.iloc[5000:, : -1].values, dtype=np.float)
    test_Y = np.array(df.iloc[5000:, -1].values, dtype=np.float32)
    test_Y = test_Y.reshape(test_Y.shape[0], 1)

    return train_X, train_Y, test_X, test_Y



def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # FPE = tf.multi(tf.subtract(prediction, y))
    # FNE = tf.constant(0.0)
    # cost = tf.add(tf.square(FPE), tf.square(FNE))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # diff = tf.subtract(prediction, y)
    # cost = tf.reduce_mean(tf.multiply(diff, diff))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    hm_epochs = 10
    # saver = tf.train.Saver()
    train_X, train_Y, test_X, test_Y = read_from_csv()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_X):
                start = i
                end = i + batch_size
                batch_x = train_X[start: end]
                batch_y = train_Y[start: end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy:', accuracy.eval({x: test_X, y: test_Y})
        # saver.save(sess, "./logs/1/model.cpkt")


if __name__ == '__main__':
    train_neural_network(x)

