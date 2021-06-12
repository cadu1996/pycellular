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

def record(title="teste.mp4",videodims=(1920,1080), fps=10, generation=100):
    ### Inicialization Record Video ###
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                        ## Define Codec to MP4
    video = cv2.VideoWriter(title, fourcc, fps, videodims)

    bgr = np.zeros(grid.shape, dtype=np.uint8)
    ### Loop to record frame in video ###
    for i in range(generation):
        next_generation()                                      
        
        img = np.dstack((grid*255, bgr, bgr))

        img = cv2.resize(img, videodims)                            ## Resize image to 1920x1080

        ### Write generation in video ###
        font = cv2.FONT_HERSHEY_SIMPLEX             
        cv2.putText(img, "generation: {0}".format(i+1),(0,100), font, 2, (255,255,255), 2, cv2.LINE_AA)

        ### Write image in video ###
        video.write(img)
    
    ### Save video ###
    video.release()