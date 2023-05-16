import numpy as np
from scipy.constants import physical_constants

s_x = np.array([[0,   1], [ 1,  0]])/2
s_y = np.array([[0, -1j], [1j,  0]])/2
s_z = np.array([[1,   0], [ 0, -1]])/2

S_x = np.array([[0,   1, 0], [ 1, 0,   1], [0,  1,  0]])/np.sqrt(2)
S_y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j,  0]])/np.sqrt(2)
S_z = np.array([[1,   0, 0], [ 0, 0,   0], [0,  0, -1]])

sigmas = {0.5: np.array([s_x, s_y, s_z]), 1: np.array([S_x, S_y, S_z])}

D = 2*np.pi*2.87e9
gamma_e =  physical_constants['electron gyromag. ratio'][0]
