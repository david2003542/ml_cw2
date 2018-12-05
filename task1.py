import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

prop = 0.5
varSigma = 0.8
image = imread('images/pug_binary.jpg')
m = image.shape[0]
n = image.shape[1]
image = image/255.0 
#this list is [h, beta,eta]
const_list = [0.2,0.4,0.3]



def add_gaussian_noise(image, prop, varSigma):
    N = int(np.round(np.prod(image.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(image.shape))[1:N], image.shape)
    e = varSigma * np.random.random(np.prod(image.shape)).reshape(image.shape)
    image2 = np.copy(image).astype('float')
    image2[index] += e[index]
    return image2

def get_neighbours(i,j,M,N,size=4):
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

def local_energy(i, j, visable_image, hidden_image):# h* \sum(x_i) -beta* \sum(x_i x_j) -eta \sum(x_i y_i)
    global m ,n, const_list
    neighbours = get_neighbours(i, j, m, n, 4)
    total_pixels = m*n
    h = const_list[0]
    beta = const_list[1]
    eta = const_list[2]
    x_sum = hidden_image[i,j]
    x_neighbord_sum = hidden_image[i][j] * np.sum((list(map(lambda x: hidden_image[x[0],x[1]], neighbours))))
    x_y_sum = visable_image[i][j] *hidden_image[i][j]
    energy = (h * x_sum - beta * x_neighbord_sum - eta *x_y_sum) / total_pixels
    return energy

def calculate_total_energy(visable_image, hidden_image2):
    global m, n 
    energy = 0.
    for i in range(m):
        for j in range(n):
            energy += local_energy(i, j, visable_image, hidden_image2)
    return energy


def icm(i, j, total_energy, visable_image, hidden_image2):
    energy = local_energy(i, j, visable_image, hidden_image2)
    other_energy = total_energy - energy
    #flip
    temp_hidden_image = np.copy(hidden_image2)
    if(temp_hidden_image[i][j]<1):
        temp_hidden_image[i][j] = -1
    else:
        temp_hidden_image[i][j] = 1
    temp_energy = local_energy(i, j, visable_image, temp_hidden_image)
    if energy > temp_energy:
        if(hidden_image2[i][j]<1):
            hidden_image2[i][j] = -1
        else:
            hidden_image2[i][j] = 1
        flipped_energy = local_energy(i, j, visable_image, hidden_image2)  
        total_energy = other_energy + flipped_energy

image_noise = add_gaussian_noise(image, prop, varSigma)
hidden_image = np.copy(image_noise)
hidden_image2 = np.copy(image_noise)
plt.imshow(hidden_image2, cmap='gray')
plt.savefig('result/noise.png')
total_energy = calculate_total_energy(image, hidden_image2)
for times in range(10):
    for i in range(m):
        for j in range(n):
            icm(i, j, total_energy, image, hidden_image2)
    plt.imshow(hidden_image2, cmap='gray')
    plt.savefig('result/restore'+str(times)+'.png')

