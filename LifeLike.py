from scipy.signal import correlate2d
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Rules to Conway's Life-like
"""
born = [3]
survival = [2,3]

"""
Space at the life elvove, by default the space have 270 x 480 pixels 
"""
grid = np.random.randint(2, size=(270,480), dtype=np.uint8)


def next_generation():
    global grid

    ### Create a kernel ###
    kernel = np.ones((3,3))
    kernel[1,1] = 0

    ### Apply Kernel in Grid ###
    neighbours = correlate2d(grid, kernel, 'same')

    ### Apply Rules in grid ###
    old_grid = grid.copy()
    grid[(old_grid == 1) & (~np.isin(neighbours,survival))] = 0
    grid[(old_grid == 0) & (np.isin(neighbours,born))] = 1

    return grid