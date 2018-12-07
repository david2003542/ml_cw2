import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import math

prop = 0.7
varSigma = 0.5
image = imread('images/pug_binary.jpg')
M = image.shape[0]
N = image.shape[1]
image = image/255.0 * 2 -1

def get_neighbours(i, j, size=4):
    global M, N
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        if (i==0 and j==0):
            n=[(0,1), (1,0), (1,1)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1), (1, N-2)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0), (M-2,1)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1), (M-2,N-2)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j), (1,j-1), (1,j+1)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j), (M-2,j-1), (M-2,j+1)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1), (i+1,1), (i-1,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2), (i-1,N-2), (i+1,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1), (i+1,j+1), (i-1,j-1), (i-1,j+1), (i+1,j-1)]
        return n

def get_gibbs_probability(neighbour_value, observed_value):
    energy =  neighbour_value
    probx_given_x1 = 1 / (1 + math.exp(energy))
    probx_given_x2 = 1 - probx_given_x1
    mean = [0,0]
    cov = [[1, 0], [0, 1]]
    proby_given_x1 = (1 / (2 * math.pi)) * math.exp(1/2*(observed_value-1)**2)
    proby_given_x2 = (1 / (2 * math.pi)) * math.exp(1/2*(observed_value+1)**2)
    probability = proby_given_x1*probx_given_x1/proby_given_x1*probx_given_x1+proby_given_x2*probx_given_x2
    return probability

def gibbs_sampling_ising():
    for times in range(1):
        for i in range(M):
            for j in range(N):
                u = np.random.uniform()
                probability = gibbs_calculate()
                if probability > u:
                    x = 1
                else:
                    x = -1
    return x


def add_gaussian_noise(image, prop, varSigma):
    N = int(np.round(np.prod(image.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(image.shape))[1:N], image.shape)
    e = varSigma * np.random.randn(np.prod(image.shape)).reshape(image.shape)
    image2 = np.copy(image).astype('float')
    image2[index] += e[index]
    return image2

def get_neighbours(i, j, size=4):
    global M, N
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        if (i==0 and j==0):
            n=[(0,1), (1,0), (1,1)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1), (1, N-2)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0), (M-2,1)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1), (M-2,N-2)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j), (1,j-1), (1,j+1)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j), (M-2,j-1), (M-2,j+1)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1), (i+1,1), (i-1,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2), (i-1,N-2), (i+1,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1), (i+1,j+1), (i-1,j-1), (i-1,j+1), (i+1,j-1)]
        return n


def icm(i, j, observed_image, drawed_image):
    pixel_change = 0
    neighbours = get_neighbours(i, j, 4)
    neighbours_sum = 0
    for neighbour in neighbours:
        neighbours_sum += drawed_image[neighbour[0]][neighbour[1]]
    observed_value = observed_image[i][j]
    prob = get_gibbs_probability(neighbours_sum, observed_value)
    #flip
    old_pixel = drawed_image[i][j]
    t = np.random.uniform(-1,1,1)
    if t < prob:
        drawed_image[i][j] = 1
    else:
        drawed_image[i][j] = -1
    if old_pixel != drawed_image[i][j]:
        pixel_change = 1
    return pixel_change



if __name__ == "__main__":
    image_noise = add_gaussian_noise(image, prop, varSigma)
    observed_image = np.copy(image_noise)
    drawed_image = np.copy(image_noise)
    for times in range(10):
        pixel_change = 0
        for i in range(M):
            for j in range(N):
                pixel_change += icm(i, j, observed_image, drawed_image)
        if pixel_change==0:
            print(times+1)
            
    plt.imshow(drawed_image, cmap='gray')
    plt.savefig('result/task2/restore2.png')