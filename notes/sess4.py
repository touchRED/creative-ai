import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipyd
from libs import gif, nb_utils, utils
from skimage.data import coffee
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

sess = tf.Session()

from libs import inception
net = inception.get_inception_model()

# import the inception graph, use that as our default graph
tf.import_graph_def(net['graph_def'], name='inception')
g = tf.get_default_graph()

# store all the operations for enumeration later
names = [op.name for op in g.get_operations()]

# get the input and output of the net
# (first and last indices of the graph)
input_name = names[0] + ':0'
x = g.get_tensor_by_name(input_name)
softmax = g.get_tensor_by_name(names[-1] + ':0')

# examine how the inception net downsizes the og img -
# crops it to a square about half the og size
og = coffee()
img = inception.preprocess(og)
print("og shape: ", og.shape) # (400, 600, 3)
print("preprocessed shape: ", img.shape) # (299, 299, 3)

# add an axis to our img to make it 4d for convolution
# in this case, we add a batch dimension
img_4d = img[np.newaxis]

def predict():
    # Predicting With the Inception Network

    # run the whole net on our image
    res = np.squeeze(sess.run(softmax, feed_dict={
        x: img_4d
    }))

    print(res.shape)

    # get the id of our top predtion, so we can get the label
    id_pred = sess.run(tf.reduce_mean(tf.argmax(res, 1)))

    # reduce the dimensionality a bit
    res = np.mean(res, 0)

    # print out the top prediction with its label!
    print(res[id_pred], net['labels'][id_pred])

    # res = np.mean(res, 0)
    # res = res / np.sum(res)
    #
    # print([(res[idx], net['labels'][idx])
    #     for idx in res.argsort()[-5:][::-1]])

def filters():
    # visualization of the first layer of convolutional filters

    # get the weights of the first conv layer (conv2d0_w)
    W = g.get_tensor_by_name('inception/conv2d0_w:0')
    W_eval = sess.run(W)

    # look at the shape
    # (7,7,3,64)
    # 7x7 size, 3 input channels, 64 output channels
    print(W_eval.shape)

    # view a montage of the filters as they are, 1 channel
    W_montage = utils.montage_filters(W_eval)
    plt.imshow(W_montage, interpolation='nearest')

    # next, stack them on top of one another
    # to view them as rgb filters
    Ws = [utils.montage_filters(W_eval[:,:,[i],:]) for i in range(3)]

    # convert to np array
    Ws = np.array(Ws)

    # during the stack, our channel dimension got set up
    # as the first dim, so we need to roll it to the last
    print("before roll:", Ws.shape)
    Ws = np.rollaxis(np.array(Ws), 0, 3)
    print("after roll:", Ws.shape)

    plt.figure()
    plt.imshow(Ws, interpolation='nearest')

    # normalize to remove some of the noise
    Ws = (Ws / np.max(np.abs(Ws)) * 128 + 128).astype(np.uint8)

    plt.figure()
    plt.imshow(Ws, interpolation='nearest')

def features():

    # retrieve our weights once again, stack them, etc
    W = g.get_tensor_by_name('inception/conv2d0_w:0')
    W_eval = sess.run(W)
    Ws = [utils.montage_filters(W_eval[:,:,[i],:]) for i in range(3)]
    Ws = np.rollaxis(np.array(Ws), 0, 3)
    Ws = (Ws / np.max(np.abs(Ws)) * 128 + 128).astype(np.uint8)

    # then, grab the output of the convolution
    feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')

    # we can examine the shape: (1, 150, 150, 64)
    # the input image was (1, 299, 299, 3)
    # it got sized in half, because of the conv2d strides
    # but it now contains 64 new channels of info
    layer_shape = sess.run(tf.shape(feature), feed_dict={x: img_4d})
    print(layer_shape)

    # we can run it to see the output of the convolution
    f = sess.run(feature, feed_dict={
        x: img_4d
    })

    # f.shape = (1, 150, 150, 64)
    # f[0].shape = (150, 150, 64)
    # after expand_dims = (150, 150, 64, 1)
    # after rollaxis = (150, 150, 1, 64)
    # that's how we get to kernel/filter dimensions
    # such that we can view each of these new channels
    montage = utils.montage_filters(np.rollaxis(np.expand_dims(f[0], 3), 3, 2))

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(inception.deprocess(img))
    axs[1].imshow(Ws, interpolation='nearest')
    axs[2].imshow(montage, cmap='gray')

def normalize(img, s=0.1):
    # normalize the image range for visualization
    z = img / np.std(img)
    return np.uint8(np.clip(
        (z - z.mean()) / max(z.std(), 1e-4) * s + 0.5, 0, 1
    ) * 255)

def compute_gradient(input_p, img, layer_name, neuron_i):
    # grab the layer
    feature = g.get_tensor_by_name(layer_name)
    # compute the gradient of the given neuron_i
    # w.r.t the given input_p
    gradient = tf.gradients(tf.reduce_mean(feature[:,:,:,neuron_i]), input_p)
    # eval the gradient and return it
    res = sess.run(gradient, feed_dict={input_p: img})[0]
    return res

def compute_gradients(input_p, img, layer_name):
    # call compute_gradient across all neurons in a layer
    # return array of gradients
    feature = g.get_tensor_by_name(layer_name)
    layer_shape = sess.run(tf.shape(feature), feed_dict={input_p: img})
    gradients = []
    for neuron_i in range(layer_shape[-1]):
        gradients.append(compute_gradient(input_p, img, layer_name, neuron_i))
    return gradients

def gradient():
    # grab the first conv layer
    feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')

    # check the gradient, i.e how it changes w.r.t input x
    # reduce to the max of the 4th channel to get the most info
    gradient = tf.gradients(tf.reduce_max(feature, 3), x)

    # tf.gradients returns a vector of partial derivatives
    # length is just 1, so we tack [0] onto the end
    res = sess.run(gradient, feed_dict={
        x: img_4d
    })[0]

    # normalize to remove noise
    r = normalize(res)

    # show the input image alongside the gradient
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(inception.deprocess(img))
    axs[1].imshow(r[0])

def gradients():
    # grab the second layer of conv, compute the gradients
    gradients = compute_gradients(x, img_4d, 'inception/conv2d1_pre_relu:0')
    # normalize the gradients
    gradients_norm = [normalize(gradient_i[0]) for gradient_i in gradients]
    montage = utils.montage(np.array(gradients_norm))

    plt.imshow(montage)

def pooling():
    # grab every max pool layer
    features = [name for name in names if 'maxpool' in name.split('/')[-1]]
    print(features)

    n_plots = len(features) + 1
    fig, axs = plt.subplots(1, n_plots, figsize=(10, 5))
    base = img_4d
    # show the original img
    axs[0].imshow(inception.deprocess(img))
    for feature_i, feature_name in enumerate(features):
        # for each maxpool layer, find the max neuron...
        feature = g.get_tensor_by_name(feature_name + ':0')
        neuron = tf.reduce_max(feature, len(feature.get_shape()) - 1)
        # compute the gradient...
        gradient = tf.gradients(tf.reduce_sum(neuron), x)
        res_i = sess.run(gradient[0], feed_dict={x: base})[0]
        # display
        axs[feature_i+1].imshow(normalize(res_i))

def dream():
    # global img_4d

    n_iterations = 50
    step = 1.0
    gif_step = 10

    # set up & normalize our img
    base = img_4d
    base = base / np.max(base)

    features = [name for name in names if 'maxpool' in name.split('/')[-1]]
    for feature_i in features:
        # grab layer
        layer = g.get_tensor_by_name(feature_i + ':0')
        # neuron = tf.reduce_max(layer, len(layer.get_shape()) - 1)
        # form gradient operation
        gradient = tf.gradients(tf.reduce_mean(layer), x)
        img_copy = base.copy()
        imgs = []

        for it_i in range(n_iterations):
            # 50 times, add the gradient back into the img
            # then run the gradient again, so on
            print(it_i, end=', ')
            res_i = sess.run(gradient[0], feed_dict={
                x: img_copy
            })[0]

            # normalize response
            res_i /= (np.max(np.abs(res_i)) + 1e-8)

            img_copy += res_i * step

            if it_i % gif_step == 0:
                imgs.append(normalize(img_copy[0]))

        for it_j in range(10):
            imgs.append(imgs[len(imgs) - 1])

        gif.build_gif(imgs, saveto='hallucination' + feature_i.split('/')[-1] + '.gif')

def noise_dream():
    img_noise = inception.preprocess(
        (np.random.randint(100, 150, size=(299, 299, 3), dtype=np.uint8))
    )[np.newaxis]

    print(img_noise.min(), img_noise.max())

    layer = g.get_tensor_by_name('inception/mixed5b_pool_reduce_pre_relu:0')
    layer_shape = sess.run(tf.shape(layer), feed_dict={
        x: img_4d
    })

    # number of output channel neurons
    n_els = layer_shape[-1]

    # reduce to the max neuron
    neuron = tf.reduce_max(layer, len(layer.get_shape())-1)
    gradient = tf.gradients(tf.reduce_mean(neuron), x)

    n_iterations = 100
    step = 1.0
    gif_step = 10
    for i in range(1):
        neuron_i = np.random.randint(n_els)
        layer_vec = np.zeros(layer_shape)
        layer_vec[..., neuron_i] = 1
        img_copy = img_noise.copy() / 255.0
        imgs = []
        for it_i in range(n_iterations):
            print(it_i, end=', ')
            res_i = sess.run(gradient[0], feed_dict={
                x: img_copy,
                layer: layer_vec
            })[0]

            res_i /= (np.max(np.abs(res_i)) + 1e-8)

            img_copy += res_i * step

            if it_i % gif_step == 0:
                imgs.append(normalize(img_copy[0]))

        gif.build_gif(imgs, saveto='noise' + str(i) + '.gif')

def objective_dream():
    img_noise = inception.preprocess(
        (np.random.randint(100, 150, size=(299, 299, 3), dtype=np.uint8))
    )[np.newaxis]

    print(img_noise.min(), img_noise.max())

    layer = g.get_tensor_by_name(names[-1] + ':0')

    # get the layer shape so we can make our zeros layer
    layer_shape = sess.run(tf.shape(layer), feed_dict={
        x: img_noise
    })

    # number of output channel neurons
    n_els = layer_shape[-1]

    # reduce to the max neuron
    # neuron = tf.reduce_max(layer, len(layer.get_shape())-1)
    # we don't need to create a neuron because we're going to
    # pass in a layer with only one active neuron
    # hence tf.reduce_max()
    gradient = tf.gradients(tf.reduce_max(layer), x)

    # pick the neuron we want to activate
    neuron_i = 323 # banana

    # gaussian_filter params
    sigma = 1.0
    blur_step = 5

    # gradient decay
    decay = 0.95

    # gradient clipping
    pth = 5

    # infinite zoom
    crop = 1

    # create a layer of zeros with just one active neuron
    layer_vec = np.zeros(layer_shape)
    layer_vec[..., neuron_i] = 1


    img_copy = img_noise.copy()
    # params for resizing/zooming
    n_img, height, width, ch = img_copy.shape
    imgs = []

    n_iterations = 1000
    step = 1.0
    gif_step = 10
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        # run the gradient, this time providing our
        # empty layer with one activated neuron
        res_i = sess.run(gradient[0], feed_dict={
            x: img_copy,
            layer: layer_vec
        })[0]

        # normalize the gradient
        res_i /= (np.max(np.abs(res_i)) + 1e-8)

        # add to image
        img_copy += res_i * step

        # decay image to reduce range of values
        # this gives a clearer picture
        img_copy *= decay

        # blur the gradient every 5 iterations
        if it_i % blur_step == 0:
            for ch_i in range(3):
                # smooth all 3 channels with a gaussian blur
                img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)

        # clip the gradient's activation
        mask = (abs(img_copy) < np.percentile(abs(img_copy), pth))
        img_copy = img_copy - (img_copy * mask)

        # crop 1x1 pixel border out of the image...
        img_copy = img_copy[:, crop:-crop, crop:-crop, :]

        # ... and resize, zooming in
        img_copy = resize(img_copy[0], (height, width), order=3,
            clip=False, preserve_range=True)[np.newaxis].astype(np.float32)

        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))

    gif.build_gif(imgs, saveto='objective.gif')
