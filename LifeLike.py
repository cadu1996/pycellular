"""

Author: Carlos Eduardo de Souza

"""



from scipy.signal import correlate2d
import cv2
import numpy as np
import matplotlib.pyplot as plt

class LifeLike():
    def __init__(self, B=[3], S=[2,3]):
        ### Rules Survival and Born ###
        self.B = B  
        self.S = S

        ### Generation Matrix ###
        self.grid = np.random.randint(2, size=(270,480), dtype=np.uint8)
        
    def next_generation(self):
        ### Create a kernel ###
        kernel = np.ones((3,3))
        kernel[1,1] = 0

        ### Apply Kernel in Grid ###
        neighbours = correlate2d(self.grid, kernel, 'same')

        ### Apply Rules in grid ###
        old_grid = self.grid.copy()
        self.grid[(old_grid == 1) & (~np.isin(neighbours,self.S))] = 0
        self.grid[(old_grid == 0) & (np.isin(neighbours,self.B))] = 1

    def write_video(self, title="teste.mp4",videodims=(1920,1080), fps=10, generation=100):
        ### Inicialization Record Video ###
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                        ## Define Codec to MP4
        video = cv2.VideoWriter(title, fourcc, fps, videodims)

        bgr = np.zeros(self.grid.shape, dtype=np.uint8)
        ### Loop to record frame in video ###
        for i in range(generation):
            self.next_generation()                                      
            
            img = np.dstack((self.grid*255, bgr, bgr))

            img = cv2.resize(img, videodims)                            ## Resize image to 1920x1080

            ### Write generation in video ###
            font = cv2.FONT_HERSHEY_SIMPLEX             
            cv2.putText(img, "generation: {0}".format(i+1),(0,100), font, 2, (255,255,255), 2, cv2.LINE_AA)
    
            ### Write image in video ###
            video.write(img)
        
        ### Save video ###
        video.release()
    
    def capture_temporal_sequence(self, generation_end=1000, interval=3, imgdims=(1920,1080)):
        bgr = np.zeros(self.grid.shape, dtype=np.uint8)
        for i in range(generation_end):
            self.next_generation()                                      
            
            #if (i == generation_end-interval-4):
            #    frame1 = self.grid.copy()
            
            #elif (i == generation_end-interval-1):
            #    frame2 = self.grid.copy()
            
            #elif (i == generation_end-1):
            #    frame3 = self.grid.copy()

            img = cv2.resize(self.grid, imgdims)                            ## Resize image to 1920x1080

            if (i == 0):
                plt.imsave("#0.png", img)
            
            elif (i == 10):
                plt.imsave("#10.png", img)
            
            elif (i == 20):
                plt.imsave("#20.png", img)
            
            elif (i == 30):
                plt.imsave("#30.png", img)

        #img = np.dstack((frame1*255, frame2*255, frame3*255))

        #img = cv2.resize(img, imgdims)                            ## Resize image to 1920x1080

        #plt.imsave("teste.png", img)
        

    def main(self):
        self.capture_temporal_sequence()

if __name__ == '__main__':
    run = LifeLike(B=[3,6,7,8], S=[3,4,6,7,8])
    run.main()
