# imports
# %pylab
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from skimage.data import astronaut
from scipy.misc import imresize
from mpl_toolkits.mplot3d import axes3d
# from matplotlib import cm

cat = imresize(astronaut(), (128, 128))
plt.style.use('ggplot')

def illustrate_minima():
    # create a figure to display our cost function
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    # x vals
    x = np.linspace(-1, 1, 200)

    # sine frequency
    hz = 10

    # cost function definition
    cost = np.sin(hz*x)*np.exp(-x)

    ax.plot(x, cost)
    ax.set_ylabel('Cost')
    ax.set_xlabel('Param')

    # since we know our cost function,
    # we can just calculate the gradient this way
    gradient = np.diff(cost)


    n_iterations = 500
    cmap = plt.get_cmap('coolwarm')
    c_norm = colors.Normalize(vmin=0, vmax=n_iterations)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
    init_p = np.random.randint(len(x)*0.2, len(x)*0.8)
    learning_rate = 1.0

    for iter_i in range(n_iterations):
        init_p -= learning_rate * gradient[int(init_p)]
        ax.plot(x[int(init_p)], cost[int(init_p)], 'ro', alpha=(iter_i + 1)/n_iterations, color=scalar_map.to_rgba(iter_i))

def illustrate_minima_3d():
    # illustrating the problem of using local minima
    # in a multivariable equation, where vastly more
    # local minima exist
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    hz = 10
    x, y = np.mgrid[-1:1:0.02, -1:1:0.02]
    X, Y, Z = x, y, np.sin(hz*x)*np.exp(-x)*np.cos(hz*y)*np.exp(-y)
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.75, cmap='jet', shade=False)
    ax.set_xlabel('Param1')
    ax.set_ylabel('Param2')
    ax.set_zlabel('Cost')

def illustrate_learning_rate():
    # illustrating the issue of learning rate,
    # and how using a large learning rate can
    # lead to overshooting the minima
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for rate_i, learning_rate in enumerate([0.01, 1.0, 500.0]):
        ax = axs[rate_i]
        x = np.linspace(-1, 1, 200)
        hz = 10
        cost = np.sin(hz*x)*np.exp(-x)
        gradient = np.diff(cost)
        ax.plot(x, cost)
        ax.set_ylabel('Cost')
        ax.set_xlabel('Param')
        ax.set_title(str(learning_rate))
        n_iterations = 500
        cmap = plt.get_cmap('coolwarm')
        c_norm = colors.Normalize(vmin=0, vmax=n_iterations)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
        init_p = np.random.randint(len(x)*0.2, len(x)*0.8)
        for iter_i in range(n_iterations):
            init_p -= learning_rate * gradient[int(init_p)]
            ax.plot(x[int(init_p)], cost[int(init_p)], 'ro', alpha=(iter_i + 1)/n_iterations, color=scalar_map.to_rgba(iter_i))

def distance(p1, p2):
    # this function is used in our cost function
    # it measures the dist between our net's prediction
    # and the actual value/label
    return tf.abs(p1 - p2)

# let's create a neural net!
# first, example data - noisy sine wave

n_observations = 1000
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

# the above code creates a noisy representation of a
# sine wave. we want our neural net to figure out that
# the underlying rep is actually just a sine wave

def neural_net(n_iterations=100, batch_size=200, learning_rate=0.02):

    # this function creates a shallow
    # let's create a prediction function and train the net!

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    # unlike placeholders, tf Variables are defined with an
    # initial value instead of being passed a value at runtime
    W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name='weight')
    B = tf.Variable(tf.constant([0], dtype=tf.float32), name='bias')

    # now we're ready to define our prediction function
    Y_pred = X * W + B

    train(X, Y, Y_pred, n_iterations=n_iterations, batch_size=batch_size, learning_rate=learning_rate)

def better_neural_net(n_iterations=100, batch_size=200, learning_rate=0.02):

    # we can improve the neural_net function by adding
    # a nonlinearity, in this case the tanh activation function

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    # we also make our network a little wider using neurons
    n_neurons = 10
    W = tf.Variable(tf.random_normal([1, n_neurons]), name='W')
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]), name='b')
    h = tf.nn.tanh(tf.matmul(tf.expand_dims(X, 1), W) + b, name='h')

    # we reduce the sum of our activation function across the neuron dimension
    Y_pred = tf.reduce_sum(h, 1)

    train(X, Y, Y_pred, n_iterations=n_iterations, batch_size=batch_size, learning_rate=learning_rate)

def train(X, Y, Y_pred, n_iterations=100, batch_size=200, learning_rate=0.02):
    # print(n_iterations)

    cost = tf.reduce_mean(distance(Y_pred, Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # plot our training data
    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, alpha=0.15, marker='+')

    with tf.Session() as sess:
        # initialize our tf Variables
        sess.run(tf.global_variables_initializer())

        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                # run our optimizer
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

            # update our cost accordingly
            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if it_i % 10 == 0:
                # every 10 iterations, update our y prediction
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)

                # plot the current trend
                ax.plot(xs, ys_pred, 'k', alpha=it_i / n_iterations)

                print(training_cost)

            # if we aren't making much headway, stop trying
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                break

            prev_training_cost = training_cost
    fig.show()
    plt.draw()

# Painting An Image

xs = []
ys = []

# loop through our image, placing input
# coordinates in xs, and pixel values in ys
for row_i in range(cat.shape[0]):
    for col_i in range(cat.shape[1]):
        xs.append([row_i, col_i])
        ys.append(cat[row_i, col_i])

xs = np.array(xs)
ys = np.array(ys)

# Normalize the inputs
xs = (xs - np.mean(xs)) / np.std(xs)

def linear(X, n_input, n_output, activation=None, scope=None):
    # define a layer of neurons
    with tf.variable_scope(scope or "linear"):
        # since we usually define a scope for each layer,
        # we can use tf.get_variable()
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output], # inputs * outputs
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output], # equal to num neurons
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h

def image_paint():

    X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

    # array representing the number of neurons per layer
    # 2 neurons for input (x,y coords), 3 for output (rgb)
    n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

    # recursively define the network
    current_input = X
    for layer_i in range(1, len(n_neurons)):
        current_input = linear(
            X=current_input,
            n_input=n_neurons[layer_i - 1],
            n_output=n_neurons[layer_i],
            activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
            scope='layer_' + str(layer_i))
    # set our prediction function to the resulting deep net
    Y_pred = current_input

    # train
    train_image(X, Y, Y_pred)

def train_image(X, Y, Y_pred, n_iterations=5000, batch_size=50, learning_rate=0.001):

    cost = tf.reduce_mean(tf.reduce_sum(distance(Y_pred, Y), 1))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        # initialize our tf Variables
        sess.run(tf.global_variables_initializer())

        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                # run our optimizer
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

            # update our cost accordingly
            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if (it_i + 1) % 20 == 0:
                # every 10 iterations, update our y prediction
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
                print(training_cost)

            # if we aren't making much headway, stop trying
            if training_cost < 40:
                fig, ax = plt.subplots(1,1)
                img = np.clip(ys_pred.reshape(cat.shape), 0, 255).astype(np.uint8)
                plt.cla()
                plt.imshow(img)
                plt.show()
                return

            # prev_training_cost = training_cost
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        fig, ax = plt.subplots(1,1)
        img = np.clip(ys_pred.reshape(cat.shape), 0, 255).astype(np.uint8)
        # plt.cla()
        plt.imshow(img)
        # plt.show()
