from skimage.io import imread
from imageformat import RGB2YUV,RGB2LAB
import numpy as np
import os
import matplotlib.pyplot as plt
from imageformat import *

def get_input(path):
    img = imread(path)
    return img

def get_output(img):
    return img[:,:,1:3]

def preprocess_input(img):
    #img2 = img /255.0
    return RGB2LAB(img)


def image_generator(files,batch_size = 128):
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files,size = batch_size)
        batch_input = []
        batch_output = [] 
         
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            inputimg = get_input(input_path )    
            inputimg = preprocess_input(inputimg)
            output = get_output(inputimg)
            inputimg = inputimg[:,:,0]
            inputimg = np.reshape(inputimg,(inputimg.shape[0],inputimg.shape[1],-1))
            batch_input += [ inputimg ]
            batch_output += [ output ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        yield( (batch_x, batch_y) )


'''
def image_generator(files,batch_size = 128):
    arr = np.array([imread(x) for x in files])
    imgs = []
    for el in arr:
        imgs.append(preprocess_input(el))
    imgs = np.array(imgs)
    inputs = imgs[:,:,:,0]
    inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],inputs.shape[2],-1))
    outputs = imgs[:,:,:,1:3]
    while True:
        # Select files (paths/indices) for the batch
        rix = np.random.randint(0,imgs.shape[0],batch_size)
        batch_x = inputs[rix]
        batch_y = outputs[rix]
        #plt.figure()
        #plt.title("LOADER")
        #plt.imshow(LAB2RGB(np.concatenate((batch_x[0],batch_y[0]),axis=2)))
        #plt.show()
        #plt.close()
        yield( (batch_x, batch_y) )
'''
def get_all_files(parent,ext):
    files = []
    for r,d,f in os.walk(parent):
        for file in f:
            if ext in file:
                files.append(os.path.join(r,file))
    return files