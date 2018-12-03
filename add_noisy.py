import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

def add_gaussian_noise(image, prop, varSigma):
    N = int(np.round(np.prod(image.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(image.shape))[1:N], image.shape)
    e = varSigma * np.random.random(np.prod(image.shape)).reshape(image.shape)
    image2 = np.copy(image).astype('float')
    image2[index] += e[index]
    return image2

def add_saltnpeppar_noise(image, prop):
    N = int(np.round(np.prod(image.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(image.shape))[1:N], image.shape)
    image2 = np.copy(image)
    image2[index] = 1 - image2[index]
    return image2

prop = 0.7
varSigma = 0.1
image = imread('images/pug_binary.jpg')
image = image / 255
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(image, cmap='gray')

image2 = add_gaussian_noise(image, prop, varSigma)
ax2 = fig.add_subplot(132)
ax2.imshow(image2, cmap='gray')
image2 = add_saltnpeppar_noise(image, prop)
ax3 = fig.add_subplot(133)
ax3.imshow(image2, cmap='gray')
fig.savefig('foo.png')
