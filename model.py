import keras
from keras.datasets import cifar10
from keras.layers import Conv2D,Input,SeparableConv2D,Concatenate,Dropout,BatchNormalization,LeakyReLU,GlobalAveragePooling2D,Dense,Multiply,UpSampling2D,Flatten
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import regularizers
from keras.initializers import RandomNormal
from keras.losses import *
import keras.backend as K
import tensorflow as tf
GAN_FLAG = False
def g_loss(base_content,target):
    if GAN_FLAG == False:
        return mean_squared_error(base_content,target)
    else:
        return binary_crossentropy(base_content,target)
def customAct(x):
    return K.tanh(x) * 100

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

class GAN:
    def __init__(self):
        self.KERNEL_SIZE = (5,5)
        self.G_INPUT_CHANNELS = 1
        self.D_INPUT_CHANNELS = 3
        self.initializer = RandomNormal(mean=0.0, stddev=1.002)
        self.get()

    def addConvStep(self,input_layer,filters,conditioning):
        filters = int(filters)
        c1 = Concatenate()([input_layer,conditioning])
        #c1 = Multiply()([input_layer,conditioning])
        c2 = SeparableConv2D(filters,self.KERNEL_SIZE,padding='same')(c1)
        #c3 = BatchNormalization()(c2)
        c3 = LeakyReLU(0.2)(c2)
        c4 = Dropout(0.3)(c3)
        return c4
    
    def build_generator(self):
        '''
        conditioning = Input(shape=(None,None,self.G_INPUT_CHANNELS))
        input_layer = Input(shape=(None,None,1))
        filters = 32
        c1 = self.addConvStep(input_layer,filters,conditioning)
        filters = 32
        c2 = self.addConvStep(c1,filters,conditioning)
        filters = 16
        c3 = self.addConvStep(c2,filters,conditioning)
        filters = 2
        c4 = SeparableConv2D(filters,self.KERNEL_SIZE,padding='same',activation=customAct)(c3)
        model = Model([input_layer,conditioning],c4)
        print("--Generator--")
        model.summary()
        '''
        conditioning = Input(shape=(None, None, 1))
        noise = Input(shape=(None,None,1))
        hid = Concatenate()([noise,conditioning])
        hid = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(hid)
        hid = Conv2D(8, (3, 3), activation='relu', padding='same')(hid)
        hid = Conv2D(16, (3, 3), activation='relu', padding='same')(hid)
        hid = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(hid)
        hid = Conv2D(32, (3, 3), activation='relu', padding='same')(hid)
        hid =  Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(hid)
        hid = UpSampling2D((2, 2))(hid)
        hid = Conv2D(32, (3, 3), activation='relu', padding='same')(hid)
        hid = UpSampling2D((2, 2))(hid)
        hid = Conv2D(16, (3, 3), activation='relu', padding='same')(hid)
        hid = UpSampling2D((2, 2))(hid)
        hid = Conv2D(2, (3, 3), activation=customAct, padding='same')(hid)
        model = Model([noise,conditioning],hid)
        
        return model
    
    def build_discriminator(self):
        input_layer = Input(shape=(256,256,2))
        condition_layer = Input(shape=(256,256,1))
        hid = Concatenate()([input_layer,condition_layer])
        #hid = Multiply()([input_layer,condition_layer])
        hid = Conv2D(32, kernel_size=5, strides=2, padding='same',kernel_initializer=self.initializer)(input_layer)
        hid = LeakyReLU()(hid)

        #hid = Conv2D(64, kernel_size=5, strides=2, padding='same')(hid)
        #hid = LeakyReLU()(hid)

        hid = Conv2D(64, kernel_size=5, strides=2, padding='same',kernel_initializer=self.initializer)(hid)
        hid = LeakyReLU()(hid)

        hid = Flatten()(hid)
        hid = Dense(64, activation='relu')(hid)
        out = Dense(1, activation='sigmoid')(hid)
        model = Model(inputs=[input_layer, condition_layer], outputs=out)
        print("--Discriminator--")
        model.summary()
        return model

    def get(self):
        optimizer = Adam(lr=0.0003)
        d_optimizer = Adam(lr=0.0006)

        #Build and compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(d_optimizer,loss='binary_crossentropy',metrics=['acc'])

        #Build generator
        self.generator = self.build_generator()
        self.generator.compile(optimizer,loss=g_loss,metrics=['acc'])
        noise = Input(shape=(256,256,1))
        conditioning = Input(shape=(256,256,self.G_INPUT_CHANNELS))
        img = self.generator([noise,conditioning])

        #Set discriminator as non-trainable to build GAN model
        self.discriminator.trainable = False

        #Get outputs for GAN model
        real = self.discriminator([img,conditioning])
        
        #build and compile GAN model
        self.gan = Model([noise,conditioning],real)
        self.gan.compile(loss='binary_crossentropy',optimizer=optimizer)
        

