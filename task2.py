import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread


image = imread('images/pug_binary.jpg')

def gibbs_sampling_ising():
    for times in range(1):
        for i in range(M):
            for j in range(N):
                u = np.random.uniform()
                