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

def check_limit(value, limit):
    if value<0:
        value=limit-1
    if value==limit:
        value=0
    return value

def add_energy_contribution(visible_arr,hidden_arr, x_val,y_val, const_list):
    h_val = const_list[0]
    beta = const_list[1]
    eta = const_list[2]
    total_pixels = hidden_arr.shape[0]*hidden_arr.shape[1]
    energy = h_val*hidden_arr[x_val,y_val]
    energy += -eta*hidden_arr[x_val,y_val]*visible_arr[x_val,y_val]
    x_neighbor = [-1,1]
    y_neighbor = [-1,1]
    for i in x_neighbor:
        for j in y_neighbor:
            x_n = check_limit(x_val +i,hidden_arr.shape[0])
            y_n = check_limit(y_val +j, hidden_arr.shape[1])
            
            energy += -beta*hidden_arr[x_val,y_val]*hidden_arr[x_n,y_n]
    energy = energy/total_pixels
    return energy

def calculate_total_energy(visible_arr, hidden_arr, const_list):
    energy = 0.
    for i in range(visible_arr.shape[0]):
        for j in range(visible_arr.shape[1]):
            energy += add_energy_contribution(visible_arr,hidden_arr,i,j,const_list)
    return energy

def icm_single_pixel(visible_arr, hidden_arr, px_x, px_y, total_energy, const_list):
    current_energy = add_energy_contribution(visible_arr, hidden_arr,px_x,px_y, const_list)
    other_energy = total_energy - current_energy
    #flip the pixel
    new_hidden_arr = np.copy(hidden_arr)
    if hidden_arr[px_x,px_y]==1:
        new_hidden_arr[px_x,px_y]=-1
    else:
        new_hidden_arr[px_x,px_y] = 1
    flipped_energy = add_energy_contribution(visible_arr, new_hidden_arr,px_x,px_y, const_list)
    #print current_energy, flipped_energy
    if flipped_energy < current_energy:
        should_flip = True
        total_energy = other_energy + flipped_energy
        hidden_arr = new_hidden_arr
        #print percent_pixel_flipped(hidden_arr, visible_arr)
    else:
        should_flip = False
    
    return (hidden_arr,should_flip,total_energy)
    #return (should_flip, hidden_arr, total_energy)

prop = 0.7
varSigma = 0.1
image = imread('images/pug_binary.jpg')
image = image / 255

image2 = add_gaussian_noise(image, prop, varSigma)



#this list is [h, beta,eta]
const_list = [0,.1,.02]
noisy_img_arr = np.copy(image2)
hidden_image = np.copy(noisy_img_arr)
total_energy= calculate_total_energy(noisy_img_arr, hidden_image, const_list)

energy_this_round = total_energy
for sim_round in range(6):
    for i in range(hidden_image.shape[0]):
        for j in range(hidden_image.shape[1]):
            hidden_image,should_flip,total_energy = icm_single_pixel(noisy_img_arr,hidden_image,i,j, total_energy,const_list)
        #print percent_pixel_flipped(hidden_image, lena_arr)
            if (total_energy - energy_this_round) == 0:
                print("Algorithm converged")
                break
            energy_this_round = total_energy
plt.imshow(hidden_image, cmap='Greys')
plt.savefig('result/restore.png')

