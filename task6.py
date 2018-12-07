import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import math

prop = 0.7
varSigma = 0.5
image = imread('images/pug_binary.jpg')
M = image.shape[0]
N = image.shape[1]
image = image/255 * 2 -1

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


def get_prop(observed_value):
    sigma = 2
    proby_givenx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - 1),2) / (2 * math.pow(sigma,2)))
    proby_givenmx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value),2) / (2 * math.pow(sigma,2)))
    return proby_givenx, proby_givenmx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def variation(observed_image):
    global M, N
    w = 0.1
    m_array = np.copy(observed_image)
    mu = np.copy(observed_image)
    q = np.copy(observed_image)
    for times in range(1):
        for i in range(M):
            for j in range(N):
                neighbours = get_neighbours(i, j, 4)
                neighbours_sum = 0
                for neighbour in neighbours:
                    neighbours_sum += mu[neighbour[0]][neighbour[1]]
                observed_value = observed_image[i][j]
                proby_givenx, proby_givenmx = get_prop(observed_value)
                m_array[i][j] = w * neighbours_sum
                mu[i][j] = math.tanh(m_array[i][j]+1/2*(proby_givenx)-(proby_givenmx))
                q[i][j] = sigmoid(2*(m_array[i][j]+1/2*((proby_givenx)-(proby_givenmx))))
    return q



if __name__ == "__main__":
    image_noise = add_gaussian_noise(image, prop, varSigma)
    plt.imshow(image_noise, cmap='gray')
    plt.savefig('result/task6/noise.png')
    for i in range(M):
        for j in range(N):
            if image_noise[i][j] >0:
                image_noise[i][j] = 1
            else:
                image_noise[i][j] = -1
    observed_image = np.copy(image_noise)
    drawed_image = np.copy(image_noise)
    q = variation(observed_image)
    for i in range(M):
        for j in range(N):
            if q[i][j] > 0.5:
                drawed_image[i][j] =1
            else:
                drawed_image[i][j] =-1

    print(drawed_image)
    # print(result)
    plt.imshow(drawed_image, cmap='gray')
    plt.savefig('result/task6/restore3.png')