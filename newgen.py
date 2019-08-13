from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from imageloader import *
import time
from imageformat import *
from telegram import *

def sample_images(gan,epoch,rand,text,telegram = True):
    imgs = gan.predict(np.expand_dims(rand,axis=0))
    yuv = np.concatenate((rand,imgs[0]),axis=2)

    rgb = LAB2RGB(yuv)
    fig,ax = plt.subplots()
    imshow(rgb)
    #imshow(rgb/255.0)
    fig.savefig("images/%d.png" % epoch)
    photo = open(f"images/{epoch}.png", 'rb')
    send_photo(photo)
    send_message(text)
    plt.close()

files = get_all_files(r'C:\lin\train\flower','.jpg')
gen = image_generator(files,8)

ne = 1000

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='rmsprop',loss='mse')

sample_interval = 10
epochs = 200
batch_size = 8
num_examples = len(files)
num_batches = int(num_examples/float(batch_size))
gen = image_generator(files,batch_size)
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
for epoch in range(epochs):
    start = time.time()
    for batch in range(num_batches):
        (x,real_imgs) = next(gen)
        g_loss = model.train_on_batch(x,real_imgs)
        end = time.time()
        text = "Epoch:%d [%d/%d] [G loss: %f] - %d" % (epoch,batch,num_batches,g_loss,end-start)
        print (text,end='\r',flush=True)
        if (epoch % sample_interval == 0) and (batch == (num_batches-1)):
            ridx = np.random.randint(0,x.shape[0],1)
            rand = x[ridx[0]]
            sample_images(model,epoch,rand,text,True)



