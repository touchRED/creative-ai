%pylab
import tensorflow as tf
from libs import utils
imgs = utils.get_celeb_imgs()

imgs[0].shape # (218, 178, 3)
plt.imshow(imgs[0])

# when displaying a singular channel, you need to provide a colormap
plt.imshow(imgs[0][:, :, 0], cmap='gray')

# type(imgs) = list
# type(imgs[0]) = numpy array

# you can do more with a numpy array than a python list,
data = np.array(imgs)
data.shape # (100, 218, 178, 3)

# like take the mean,
mean_img = np.mean(data, axis=0) # gotta specify an axis tho
plt.imshow(mean_img.astype(np.uint8)) # typecast to a reasonable img typecast

# and the std dev
std_img = np.std(data, axis=0) # provide axis
plt.imshow(std_img.astype(np.uint8)) # typecast

# we can also take the mean of the std_img
# across the color channel, creating a heat map
plt.imshow(np.mean(std_img, axis=2).astype(np.uint8))

# right now our data is in a 4 dimensional array
# (100, 218, 178, 3)
# let's flatten it down

flattened = data.ravel() # flatten our data to 1 dimension

# in 1 dimension, we can make a histogram to visualize the dist of color
plt.hist(flattened, 255)

# we can also do this with our images
plt.hist(mean_img.ravel(), 255)
plt.hist(std_img.ravel(), 255)

# normalization
# let's first subtract the mean from an image
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)

axs[0].hist(data[0].ravel(), 20)
axs[1].hist(mean_img.ravel(), 20)
axs[2].hist((data[0] - mean_img).ravel(), 20)

# then we can normalize by subtracting the mean and
# dividing by the stddev
axs[0].hist((data[0] - mean_img).ravel(), 20)
axs[1].hist(std_img.ravel(), 20)
axs[2].hist(((data[0] - mean_img) / std_img), 20)

#tensorflow shit

# creating a Session
# you could also do tf.InteractiveSession() in console
sess = tf.Session()

# numpy linspace vs tf linspace
x = np.linspace(-3.0, 3.0, 100)
print(x.shape)
print(x.dtype)

x = tf.linspace(-3.0, 3.0, 100)
print(x.get_shape().as_list())
print(sess.run(x)) # or x.eval(session=sess)

# print(x.eval()) in InteractiveSession

# Gaussian kernel
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
    (2.0 * tf.pow(sigma, 2.0)))) *
    (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

res = sess.run(z)
plt.plot(res) # plot 1D Gaussian curve

# now let's make this into a kernel we can use
# first we'll take it to 2D using matrix multiplication

ksize = z.get_shape().as_list()[0]

# resize z to ksize rows by 1 column, times 1 row by ksize columns
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
plt.imshow(sess.run(z_2d)) # use imshow to display the 2D kernel

# convolution

from skimage import data
img = data.camera().astype(np.float32)
plt.imshow(img, cmap='gray')
print(img.shape) # (512, 512)

# our image is 2D, but tf needs it to be in 4D form:
# Batch x Height x Width x Channel
# so we need to resize our image to:
# 1 x Height x Width x 1

img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
print(img_4d.get_shape().as_list()) # (1, 512, 512, 1)

# we also need to reshape our kernel to 4D like so:
# Kernel Height x Kernel Width x Num Input Channels x Num Output Channels

z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
print(z_4d.get_shape().as_list())

convolved = tf.nn.conv2d(img_4d, z_4d, strides=[1,1,1,1], padding='SAME')
res = sess.run(convolved)

plt.imshow(np.squeeze(res), cmap='gray')
# or
plt.imshow(res[0,:,:,0], cmap='gray')

# now we have a blurred image!

# Gabor Kernel

xs = tf.linspace(-3.0, 3.0, ksize)
ys = tf.sin(xs)

plt.plot(sess.run(ys)) # sine wave !

ys = tf.reshape(ys, [ksize, 1]) # reshape ys to a column vector

# next we'll make this 2D by multiplying by ones
ones = tf.ones((1, ksize))
wave = tf.matmul(ys, ones)
plt.imshow(sess.run(wave), cmap='gray')

gabor = tf.multiply(wave, z_2d) # create 2D Gabor kernel
plt.imshow(sess.run(gabor), cmap='gray')

# Tensorflow Placeholders
# since we're repeating many previous operations, it
# makes more sense to redefine some things as placeholders

img = tf.placeholder(tf.float32, shape=[None, None], name='img')
img_3d = tf.expand_dims(img, 2) # add singleton dim at specified axis
img_4d = tf.expand_dims(img_3d, 0)

# now our img placeholder is in correct 4D form
# 1 x W x H x 1

mean = tf.placeholder(tf.float32, name='mean')
sigma = tf.placeholder(tf.float32, name='sigma')
ksize = tf.placeholder(tf.int32, name='ksize')

x = tf.linspace(-3.0, 3.0, ksize)
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
    (2.0 * tf.pow(sigma, 2.0)))) *
    (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

z_2d = tf.matmul(
    tf.reshape(z, tf.stack([ksize, 1])),
    tf.reshape(z, tf.stack([1, ksize])))

ys = tf.sin(x)
ys = tf.reshape(ys, tf.stack([ksize, 1]))

ones = tf.ones(tf.stack([1, ksize]))

wave = tf.matmul(ys, ones)
gabor = tf.multiply(wave, z_2d)
gabor_4d = tf.reshape(gabor, tf.stack([ksize, ksize, 1, 1]))

convolved = tf.nn.conv2d(img_4d, gabor_4d, strides=[1,1,1,1], padding='SAME')
convolved_img = convolved[0,:,:,0]

# now we have an entire graph dedicated to this kernel
# we can eval convolved_img by passing in data with feed_dict()

res = sess.run(convolved_img, feed_dict={
    img: data.camera(),
    mean: 0.0,
    sigma: 1.0,
    ksize: 100
})
plt.imshow(res, cmap='gray')
