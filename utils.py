import numpy as np
import cv2
import glob
import os

def load_images(path):
    print(os.path.join(path,"*.jpg"))
    return np.array([cv2.imread(file) for file in glob.glob(os.path.join(path,"*.jpg"))])
def gen_noise(n,shape=(32,32)):
    x,y = shape
    result = [np.random.random((x,y,1)) for i in range(0,n)]
    return np.array(result)