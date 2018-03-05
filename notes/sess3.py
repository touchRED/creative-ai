import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from libs.utils import linear
from libs.utils import montage
from libs.utils import montage_filters
from libs import gif
from tensorflow.python.framework.ops import reset_default_graph
import IPython.display as ipyd
plt.style.use('ggplot')

from libs import datasets
ds = datasets.MNIST()

imgs = ds.X[:1000].reshape((-1, 28, 28))

mean_img = np.mean(ds.X, axis=0)
std_img = np.std(ds.X, axis=0)

# we're going to build our encoder as a network
# with progressively smaller layers
dimensions = [512, 256, 128, 64]

# we'll do this using matrix multiplication
# 784 input dims * a 784x512 matrix
# then 512 input dims & a 512x256 matrix
# and so on...

# number of features is equal to number of pixels in img (784)
n_features = ds.X.shape[1]

def encoder(n_epochs = 10):
    reset_default_graph()
    # fully connected model

    X = tf.placeholder(tf.float32, [None, n_features])

    current_input = X
    n_input = n_features

    # let's share weights, so we can transpose them in the decoder
    Ws = []

    # first build the encoder
    global dimensions
    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope("encoder/layer/{}".format(layer_i)):
            # create our weights for each layer
            W = tf.get_variable(
                name='W',
                shape=[n_input, n_output],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)
            )
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
            )
            # we could add a bias here, but
            # it isn't entirely necessary atm
            h = tf.matmul(current_input, W)
            h = tf.nn.bias_add(
                name='h',
                value=h,
                bias=b
            )
            current_input = tf.nn.relu(h)
            Ws.append(W) # save our weights
            n_input = n_output

    Ws = Ws[::-1] # reverse the weights for the decoder

    # reverse the dims as well, adding 784 on the end for true output
    dimensions = dimensions[::-1][1:] + [ds.X.shape[1]]

    # then the decoder
    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):
            # transpose our weight matrices so we can
            # expand our size back to 784
            W = tf.transpose(Ws[layer_i])
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
            )
            h = tf.matmul(current_input, W)
            current_input = tf.nn.relu(h)
            n_input = n_output

    Y = current_input # store our entire net in Y

    # use Y in our cost, reduce mean across both dims
    cost = tf.reduce_mean(tf.squared_difference(X, Y), 1)
    cost = tf.reduce_mean(cost)

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100

    examples = ds.X[:100]

    imgs = []
    fig, ax = plt.subplots(1, 1)

    for epoch_i in range(n_epochs):
        for batch_X, _ in ds.train.next_batch():
            # train the optimizer in stochastic batches
            sess.run(optimizer, feed_dict={X: batch_X - mean_img})
        # run the first 100 images thru the net,
        # and reconstruct them to (28, 28)
        recon = sess.run(Y, feed_dict={X: examples - mean_img})
        recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
        img_i = montage(recon).astype(np.uint8)
        imgs.append(img_i)
        ax.imshow(img_i, cmap='gray')
        fig.canvas.draw()
        print(epoch_i, sess.run(cost, feed_dict={X: batch_X - mean_img}))

    gif.build_gif(imgs, saveto='ae.gif', cmap='gray')


def convolutional():
    reset_default_graph()

    # start off with the same X placeholder we already had
    X = tf.placeholder(tf.float32, [None, n_features])

    # reshape it to a 4d tensor for use in a convolutional net
    # the -1 value tells tf to compute it such that total
    # size remains constant
    X_tensor = tf.reshape(X, [-1, 28, 28, 1])

    # similar to making a list of neurons per layer in a
    # fully connected net, we're going to make a list of
    # filters per layer in our convolutional net
    n_filters = [16, 16, 16]
    filter_sizes = [4, 4, 4] # also make a list of filter sizes (4x4)

    current_input = X_tensor
    # we have one input bc we have 1 channel
    n_input = 1

    # store weights and shapes for decoder
    Ws = []
    shapes = []

    # start making our layers
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope("encoder/layer/{}".format(layer_i)):

            # save shape for decoder
            shapes.append(current_input.get_shape().as_list())

            # create filter kernel
            W = tf.get_variable(
                name='W',
                shape=[
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input,
                    n_output
                ],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)
            )
            # convolute our input using the kernel we just made
            # TODO: look into why you use 2x2 strides here
            h = tf.nn.conv2d(current_input, W, strides=[1,2,2,1], padding='SAME')

            # activate our convolution
            current_input = tf.nn.relu(h)

            Ws.append(W)
            n_input = n_output

    # reverse everything to construct our decoder
    Ws.reverse()
    shapes.reverse() # [[None, 7, 7, 16], [None, 14, 14, 16], [None, 28, 28, 1]]
    n_filters.reverse()
    n_filters = n_filters[1:] + [1]

    # loop over reversed shapes array
    for layer_i, shape in enumerate(shapes):
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):
            # retrieve the shared weight of the opposite layer
            W = Ws[layer_i]
            # convolute the input using the transpose,
            # similarly to our fully connected net
            # and output to the right shape using stack()
            h = tf.nn.conv2d_transpose(current_input, W,
                tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
                strides=[1,2,2,1], padding='SAME')

            # activate that bihhh
            current_input = tf.nn.relu(h)

    # save our net in Y
    Y = current_input
    # reshape it to the original input dims
    Y = tf.reshape(Y, [-1, n_features])

    # construct the cost & optimizer just like we did earlier
    cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # now we can train just like before!
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    n_epochs = 5
    examples = ds.X[:100]
    imgs = []
    fig, ax = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        for batch_X, _ in ds.train.next_batch():
            sess.run(optimizer, feed_dict={X: batch_X - mean_img})
        recon = sess.run(Y, feed_dict={X: examples - mean_img})
        recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
        img_i = montage(recon).astype(np.uint8)
        imgs.append(img_i)
        ax.imshow(img_i, cmap='gray')
        fig.canvas.draw()
        print(epoch_i, sess.run(cost, feed_dict={X: batch_X - mean_img}))

    gif.build_gif(imgs, saveto='conv-ae.gif', cmap='gray')

def regression():
    reset_default_graph()
    # fully connect regression model for MNIST classification

    ds = datasets.MNIST(split=[0.8, 0.1, 0.1])

    n_input = 28*28 # 784 inputs for each pixel
    n_output = 10 # 10 outputs for one-hot encoding

    X = tf.placeholder(tf.float32, [None, n_input])
    # we have a true Y in this case,
    # since we have labels for each image
    Y = tf.placeholder(tf.float32, [None, n_output])

    # create a 1 layer network
    Y_pred, W = linear(
        x=X,
        n_output=n_output,
        activation=tf.nn.softmax,
        name='layer1'
    )

    # we'll write our loss function as a cross entropy
    # this is better suited for the purposes of classification
    cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
    # optimize it in the same way we normally would
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # take the largest number out of predicted
    # and actual outputs
    predicted_y = tf.argmax(Y_pred, 1)
    actual_y = tf.argmax(Y, 1)

    # see if our net's guess was right
    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 50
    n_epochs = 5

    # train the net
    for epoch_i in range(n_epochs):
        for batch_xs, batch_ys in ds.train.next_batch():
            sess.run(optimizer, feed_dict={
                X: batch_xs,
                Y: batch_ys
            })
        valid = ds.valid
        # run our accuracy function on valid input/output
        print(sess.run(accuracy, feed_dict={
            X: valid.images,
            Y: valid.labels
        }))
    test = ds.test
    # test our accuracy with test data
    print(sess.run(accuracy, feed_dict={
        X: test.images,
        Y: test.labels
    }))

    g = tf.get_default_graph()
    # retrieve and run the weights of our regression
    W = g.get_tensor_by_name('layer1/W:0')
    W_arr = np.array(sess.run(W))

    fig, ax = plt.subplots(1, 10, figsize=(28, 3))
    for col_i in range(W_arr.shape[1]):
        # display the weights as imgs (you can see numbers!!!)
        img = W_arr[:, col_i].reshape((28, 28))
        ax[col_i].grid()
        ax[col_i].get_yaxis().set_visible(False)
        ax[col_i].get_xaxis().set_visible(False)
        ax[col_i].imshow(img, cmap='coolwarm')

def conv_regression():
    reset_default_graph()
    # building a convolutional model for classification of MNIST

    ds = datasets.MNIST(one_hot=True, split=[0.8, 0.1, 0.1])
    # 784 inputs, 10 outputs
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    n_output = 10

    # reshape input into 4d tensor for conv2d
    X_tensor = tf.reshape(X, [-1, 28, 28, 1])

    # 5x5 filter size
    filter_size = 5

    # first we do 1 x 32
    # then 32 x 64
    # then 64 x 128
    n_filters_in = 1
    n_filters_out = 32

    # define weights for our first set of kernels
    W_1 = tf.get_variable(
        name='W',
        shape=[filter_size, filter_size, n_filters_in, n_filters_out],
        initializer=tf.random_normal_initializer()
    )

    # bias is always n_outputs in size
    b_1 = tf.get_variable(
        name='b',
        shape=[n_filters_out],
        initializer=tf.constant_initializer()
    )

    # convolve input, add bias, and activate
    # NOTE: strides = [batch, height, width, channel]
    # so this means that we're moving 2x2 across every img
    h_1 = tf.nn.conv2d(X_tensor, W_1, strides=[1,2,2,1], padding='SAME')
    h_1 = tf.nn.bias_add(h_1, b_1)
    h_1 = tf.nn.relu(h_1)

    # onto the next layer, 64 kernels
    n_filters_in = 32
    n_filters_out = 64
    W_2 = tf.get_variable(
        name='W2',
        shape=[filter_size, filter_size, n_filters_in, n_filters_out],
        initializer=tf.random_normal_initializer()
    )
    b_2 = tf.get_variable(
        name='b2',
        shape=[n_filters_out],
        initializer=tf.constant_initializer()
    )
    h_2 = tf.nn.conv2d(h_1, W_2, strides=[1,2,2,1], padding='SAME')
    h_2 = tf.nn.bias_add(h_2, b_2)
    h_2 = tf.nn.relu(h_2)

    # let's reshape so we can go through
    # to a fully connected linear layer
    h_2_flat = tf.reshape(h_2, [-1, 7*7*n_filters_out])

    h_3, W = linear(h_2_flat, 128, activation=tf.nn.relu, name='fc_1')

    # shape the last linear layer such that
    # it gives us the right amount of outputs
    # also use softmax instead of relu so we get a distribution
    Y_pred, W = linear(h_3, n_output, activation=tf.nn.softmax, name='fc_2')

    # define cross entropy and optimizer
    cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    # helpful accuracy function
    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # restore our model if it exists
    if os.path.exists("sess3_checkpoints/conv_regression.ckpt.index"):
        print("Restoring model")
        saver.restore(sess, "sess3_checkpoints/conv_regression.ckpt")

        test = ds.test
        print(sess.run(accuracy, feed_dict={
            X: test.images,
            Y: test.labels
        }))

        # retrieve our weights from the first convolutional layer
        W1 = sess.run(W_1)
        plt.figure(figsize=(10, 10))
        plt.grid()
        # display all 32 kernels as images
        plt.imshow(montage_filters(W1), cmap='coolwarm', interpolation='nearest')

        # display the second layer, too
        W2 = sess.run(W_2)
        plt.figure(figsize=(10,10))
        plt.grid()
        plt.imshow(montage_filters(W2 / np.max(W2)), cmap='coolwarm')
        return

    # if not, train
    print("training")
    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_xs, batch_ys in ds.train.next_batch():
            sess.run(optimizer, feed_dict={
                X: batch_xs,
                Y: batch_ys
            })
        valid = ds.valid
        print(sess.run(accuracy, feed_dict={
            X: valid.images,
            Y: valid.labels
        }))

    test = ds.test
    print(sess.run(accuracy, feed_dict={
        X: test.images,
        Y: test.labels
    }))

    save_path = saver.save(sess, "./sess3_checkpoints/conv_regression.ckpt")
    print("Model saved")

    # retrieve our weights from the first convolutional layer
    W1 = sess.run(W_1)
    plt.figure(figsize=(10, 10))
    plt.grid()
    # display all 32 kernels as images
    plt.imshow(montage_filters(W1), cmap='coolwarm', interpolation='nearest')

    # display the second layer, too
    # W2 = sess.run(W_2)
    # plt.figure(figsize=(10, 10))
    # plt.grid()
    # plt.imshow(montage_filters(W2 / np.max(W2)), cmap='coolwarm')
