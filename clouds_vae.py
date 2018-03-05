import os
import utils
import tensorflow as tf
import h5py
from noise import pnoise1
from keras.layers import Input, Dense, Lambda, Layer
from keras import regularizers, optimizers, objectives, metrics
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import mnist
import gif
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

sess = tf.Session()

dirname = "clouds2"

# Load every image file in the provided directory
filenames = [os.path.join(dirname, fname)
             for fname in os.listdir(dirname)]

imgs = [plt.imread(fname)[..., :3] for fname in filenames]

imgs = np.array(imgs).astype(np.float32)
imgs = imgs[:480]

# mean = np.mean(imgs, axis=0)
# std = np.std(imgs, axis=0)
# imgs = (imgs - mean) / std

np.random.shuffle(imgs)

flattened = tf.reshape(imgs, (480, 30000))

flattened = sess.run(flattened)
print(flattened.shape)

values = tf.reduce_sum(flattened, axis=1)

idxs = sess.run(tf.nn.top_k(values, k=100)[1])

sorted_imgs = np.array([imgs[idx_i] for idx_i in idxs])
# plt.imshow(utils.montage(sorted_imgs, 'sorted.png'))

# VAE time

batch_size = 20
epochs = 100
epsilon_std = 1.0
original_dim = 30000
compress_dim = 2048
intermediate_dim = 256
encoding_dim = 32
latent_dim = 2

x = Input(batch_shape=(batch_size, original_dim))
# c = Dense(compress_dim, activation='relu')(x)
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_c = Dense(compress_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
# c_decoded = decoder_c(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

def vae():
    # y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, x_decoded_mean)
    # vae.compile(optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.), loss=vae_loss)
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    x_train = flattened[100:]
    x_test = flattened[:100]

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

def save():
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    # _c_decoded = decoder_c(_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    generator.save("latent.h5")

def save_weights():
    if(os.path.exists('latent_best.h5')):
        generator = load_model('latent_best.h5')
    generator.save_weights('generator.hdf5')
    with open('generator.json', 'w') as f:
        f.write(generator.to_json())

def make_gif():
    if(os.path.exists('latent_best.h5')):
        generator = load_model('latent_best.h5')

    imgs = []
    location = [0,0]
    velocity = [0,0]
    acceleration = [0,0]
    noise = np.random.rand(2) * 10
    # print(acceleration.shape)
    for i in range(150):
        noise[0] = noise[0] + 0.05
        noise[1] = noise[1] + 0.05
        acceleration[0] = pnoise1(noise[0])
        acceleration[1] = pnoise1(noise[1])
        velocity[0] = velocity[0] + acceleration[0]
        velocity[1] = velocity[1] + acceleration[1]
        velocity = np.clip(velocity, -50, 50)
        location[0] = location[0] + velocity[0]
        location[1] = location[1] +  velocity[1]

        if(location[0] < -100 or location[0] > 100):
            velocity[0] = velocity[0] * -1
        if(location[1] < -100 or location[1] > 100):
            velocity[1] = velocity[1] * -1

        print(location[0], location[1])
        # location = np.add(velocity, location)
        z_sample = np.array([[location[0], location[1]]])
        x_decoded = generator.predict(z_sample)
        img = x_decoded[0].reshape(100, 100, 3)
        imgs.append(img)

    gif.build_gif(imgs, saveto='clout.gif')


def nums(left, right):
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    # _c_decoded = decoder_c(_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    # generator = Model(decoder_input, _x_decoded_mean)
    if(os.path.exists('latent.h5')):
        generator = load_model('latent.h5')
    else:
        generator = Model(decoder_input, _x_decoded_mean)


    n = 15
    # digit_size = 28
    digit_size = 100
    figure = np.zeros((digit_size * n, digit_size * n, 3))

    grid_x = np.linspace(left, right, n)
    grid_y = np.linspace(left, right, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # z_sample = np.array([[xi, yi]]) * epsilon_std
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size, :] = digit

    plt.figure()
    plt.imshow(figure)
    plt.show()
