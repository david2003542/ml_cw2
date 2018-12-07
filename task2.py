import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import math

prop = 0.7
varSigma = 0.5
# image = imread('images/loli_grey.png')
image = imread('images/pug_binary.jpg')
M = image.shape[0]
N = image.shape[1]
image = image/255 * 2 -1

def get_gibbs_probability(neighbour_value, observed_value):
    energy =  neighbour_value
    probx_given_x1 = 1 / (1 + math.exp(energy))
    probx_given_x2 = 1 - probx_given_x1
    sigma = 2
    proby_given_x1 = (1 / (2 * math.pi)) * math.exp(1/2*(observed_value-1)**2)
    proby_given_x2 = (1 / (2 * math.pi)) * math.exp(1/2*(observed_value+1)**2)
    proby_givenx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - 1),2) / (2 * math.pow(sigma,2)))
    proby_givenmx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value),2) / (2 * math.pow(sigma,2)))
    probability = proby_given_x1*probx_given_x1/proby_given_x1*probx_given_x1+proby_given_x2*probx_given_x2
    # print(probability)
    return probability

# def prob_gibbs(neighbour_values, observed_value):
#     index_nei = 4/len(neighbour_values)
#     sigma = 0.5
#     nei_similarity = sum(target * neighbour_values)
#     probx_givenx = 1 / (1 + math.exp(-nei_similarity * index_nei)) #sigmoid #probx = 1 / (1 + math.exp(-nei_energy)) * math.pow(0.5, len(neighbour_values))
#     probmx_givenx = 1 - probx_givenx
#     proby_givenx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - 1),2) / (2 * math.pow(sigma,2)))
#     proby_givenmx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value),2) / (2 * math.pow(sigma,2)))
#     result = (proby_givenx * probx_givenx)/(proby_givenx * probx_givenx + proby_givenmx * probmx_givenx)
#     # print(target,nei_similarity,observed_value,result)
#     return result

def gibbs_sampling_ising():
    for times in range(10):
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
    plt.imshow(image_noise, cmap='gray')
    plt.savefig('result/task2/noise.png')
    for i in range(M):
        for j in range(N):
            if image_noise[i][j] >0:
                image_noise[i][j] = 1
            else:
                image_noise[i][j] = -1
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
    plt.savefig('result/task2/restore3.png')