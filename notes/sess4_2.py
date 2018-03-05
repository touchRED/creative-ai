import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from libs import gif, utils, nb_utils
from tensorflow.python.framework.ops import reset_default_graph
from libs import vgg16 as vgg
from skimage.data import rocket

sess = tf.Session()

net = vgg.get_vgg_model()

tf.import_graph_def(net['graph_def'], name='vgg')
g = tf.get_default_graph()

names = [op.name for op in g.get_operations()]

# with tf.Session(graph=g) as sess, g.device('/cpu:0'):
#     tf.import_graph_def(net['graph_def'], name='vgg')
#     names = [op.name for op in g.get_operations()]

# nb_utils.show_graph(net['graph_def'])
# print(names)

x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')

# img_download = utils.download('http://images.metmuseum.org/CRDImages/ep/original/DT210071.jpg')
og = plt.imread("Sisley.jpg")
# plt.imshow(rocket)

img = vgg.preprocess(og)
plt.imshow(vgg.deprocess(img))

img_4d = img[np.newaxis]


content_layer = 'vgg/conv4_2/conv4_2:0'
content_features = sess.run(g.get_tensor_by_name(content_layer), feed_dict={
    x: img_4d,
    'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
    'vgg/dropout/random_uniform:0': [[1.0] * 4096]
})

print(content_features.shape)

# filepath = utils.download('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg/640px-El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg')
filepath = utils.download('http://images.metmuseum.org/CRDImages/ma/original/DT5833.jpg')
# style_og = plt.imread(filepath)[15:-15, 190:-190, :]
style_og = plt.imread(filepath)
# plt.imshow(style_og)

style_img = vgg.preprocess(style_og)
plt.figure()
plt.imshow(vgg.deprocess(style_img))
style_4d = style_img[np.newaxis]

style_layers = ['vgg/conv1_1/conv1_1:0',
                'vgg/conv2_1/conv2_1:0',
                'vgg/conv3_1/conv3_1:0',
                'vgg/conv4_1/conv4_1:0',
                'vgg/conv5_1/conv5_1:0']

style_activations = []

for style_i in style_layers:
    layer = g.get_tensor_by_name(style_i)
    style_activation_i = sess.run(layer, feed_dict={
        x: style_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
        'vgg/dropout/random_uniform:0': [[1.0] * 4096]
    })
    style_activations.append(style_activation_i)

style_features = []
for style_activation_i in style_activations:
    s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
    gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
    style_features.append(gram_matrix.astype(np.float32))

def predict_with_drop():
    res = sess.run(softmax, feed_dict={
        x: img_4d
    })[0]

    print([(res[idx], net['labels'][idx]) for idx in res.argsort()[-5:][::-1]])

def predict_no_drop():
    res = sess.run(softmax, feed_dict={
        x: img_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
        'vgg/dropout/random_uniform:0': [[1.0] * 4096]
    })[0]

    print([(res[idx], net['labels'][idx]) for idx in res.argsort()[-5:][::-1]])

def total_variation_loss(x):
    h, w = x.get_shape().as_list()[1], x.get_shape().as_list()[1]
    dx = tf.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
    dy = tf.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1:, :])
    return tf.reduce_sum(tf.pow(dx + dy, 1.25))

def style():
    reset_default_graph()
    sess = tf.Session()
    net_input = tf.Variable(img_4d)
    tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': net_input})
    g = tf.get_default_graph()
    # names = [op.name for op in g.get_operations()]
    # print(names)
    content_dist = g.get_tensor_by_name(content_layer) - content_features
    content_dist /= content_features.size
    content_loss = tf.nn.l2_loss(content_dist)

    style_loss = np.float32(0.0)
    for style_layer_i, style_gram_i in zip(style_layers, style_features):
        layer_i = g.get_tensor_by_name(style_layer_i)
        layer_shape = layer_i.get_shape().as_list()
        layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
        gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
        style_loss = tf.add(
            style_loss,
            tf.nn.l2_loss((gram_matrix - style_gram_i) / np.float32(style_gram_i.size))
        )

    tv_loss = total_variation_loss(net_input)

    loss = 0.1 * content_loss + 5.0 * style_loss + 0.01 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    sess.run(tf.global_variables_initializer())
    n_iterations = 100
    og_img = sess.run(net_input)
    imgs = []
    fig, ax = plt.subplots(1, 3, figsize=(22, 5))
    for it_i in range(n_iterations):
        _, this_loss, synth = sess.run([optimizer, loss, net_input],
            feed_dict={
                'vgg/dropout_1/random_uniform:0':
                    np.ones(g.get_tensor_by_name('vgg/dropout_1/random_uniform:0').get_shape().as_list()),
                'vgg/dropout/random_uniform:0':
                    np.ones(g.get_tensor_by_name('vgg/dropout/random_uniform:0').get_shape().as_list())
            }
        )
        print("%d: %f, (%f - %f)" %
            (it_i, this_loss, np.min(synth), np.max(synth)))

        if it_i % 1 == 0:
            imgs.append(np.clip(synth[0], 0, 1))
            # plt.cla()
            # ax[0].imshow(vgg.deprocess(img))
            # ax[1].imshow(vgg.deprocess(style_img))
            # ax[2].imshow(vgg.deprocess(synth[0]))
            # # plt.show()
            # fig.canvas.draw()

    for it_j in range(20):
        imgs.append(imgs[len(imgs) - 1])

    gif.build_gif(imgs, saveto='style-bosch2.gif')
