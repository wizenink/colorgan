import keras
import numpy as np
from keras.datasets import cifar10
import scipy.misc
from model import GAN,GAN_FLAG
from utils import gen_noise,load_images
from imageformat import *
from telegram import *
import time
from imageloader import *
import matplotlib.pyplot as plt
from os import path


class Trainer:
    
    def __init__(self):
        self.d_loss_a = []
        self.gan_loss_a = []
        self.gan = GAN()
    def get_data(self):
        (train_rgb,_),(test_rgb,_) = cifar10.load_data()
        #Convert training and test images to YUV color space
        #train_rgb = load_images(r"D:\GOOGLE DRIVE\TFG\proyecto\linnaeus\Linnaeus 5 256X256\train")
        #test_rgb = np.array([])
        train_rgb = train_rgb/255.0
        test_rgb = test_rgb/255.0
        train_yuv = RGB2YUV(train_rgb)
        test_yuv = RGB2YUV(test_rgb)

        #X is the generator input, only the luminance channel of the YUV images.
        x_train = train_yuv[:,:,:,0]
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],-1)
        x_test = test_yuv[:,:,:,0]
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],-1)

        #Y is the discriminator input, the chrominance channels of the YUV images, in order to classify them as valid or invalid.
        y_train = train_yuv[:,:,:,1:3]
        y_test = test_yuv[:,:,:,1:3]

        print("Got training data:")
        #x_train = x_train/x_train.max()
        #y_train = y_train/y_train.max()
        print("x_train{",x_train.shape,"}:[",x_train.min(),",",x_train.max(),"]")
        print("y_train{",y_train.shape,"}:[",y_train.min(),",",y_train.max(),"]")
        return (x_train,y_train),(x_test,y_test)


    def pretrain_discriminator(self,batch_size=16,epochs=100,truefiles=r'C:\lin\train\flower',fakefiles=r'D:\GOOGLE DRIVE\TFG\proyecto\fake'):

        filenames_true = get_all_files(truefiles,'.jpg')
        filenames_fake = get_all_files(fakefiles,'.png')
        num_examples = max(len(filenames_fake),len(filenames_true))
        num_batches = int(num_examples/float(batch_size))
        half_batch = int(batch_size/2)

        gen_fake = image_generator(filenames_fake,half_batch)
        gen_true = image_generator(filenames_true,half_batch)

        labels_fake = np.zeros((half_batch,1))
        labels_true = np.ones((half_batch,1))
        all_labels = np.concatenate((labels_fake,labels_true))
        for epoch in range(epochs):
            losses = []
            accuracies = []
            for batch in range(num_batches):
                (fake_x,fake_y) = next(gen_fake)
                (true_x,true_y) = next(gen_true)
                all_x = np.concatenate((fake_x,true_x))
                all_y = np.concatenate((fake_y,true_y))
                p = np.random.permutation(len(all_x))
                all_x = all_x[p]
                all_y = all_y[p]
                all_labels = all_labels[p]
                d_loss = self.gan.discriminator.train_on_batch([all_y,all_x],all_labels)
                losses.append(d_loss[0])
                accuracies.append(d_loss[1])
                text = "Epoch:%d [%d/%d] [D loss: %f, acc:%f]" % (epoch,batch,num_batches,np.mean(losses),np.mean(accuracies)*100)
                print(text,end='\r')
            print()



    def pretrain_generator(self,batch_size=8,epochs=130,sample_interval=10,num_gen=1000,files=r'C:\lin\train\flower',fake="fake/"):
        filenames = get_all_files(files,'.jpg')
        #filenames = filenames[:2000]
        num_examples = len(filenames)
        num_batches = int(num_examples/float(batch_size))
        gen = image_generator(filenames,batch_size)
        IMAGE_WIDTH = 256
        IMAGE_HEIGHT = 256
        for epoch in range(epochs):
            start = time.time()
            for batch in range(num_batches):
                (x,real_imgs) = next(gen)
                noise = gen_noise(batch_size,(IMAGE_WIDTH,IMAGE_HEIGHT))
                g_loss = self.gan.generator.train_on_batch([noise,x],real_imgs)
                end = time.time()
                text = "Epoch:%d [%d/%d] [G loss: %f, acc:%f] - %d" % (epoch,batch,num_batches,g_loss[0],g_loss[1]*100,end-start)
                print (text,end='\r',flush=True)
                if (epoch % sample_interval == 0) and (batch == (num_batches-1)):
                    ridx = np.random.randint(0,x.shape[0],1)
                    rand = x[ridx[0]]
                    self.sample_images(self.gan,epoch,rand,text,True,phase='pre')
        num_batch_gen = int(num_gen/float(batch_size))
        imnumber = 0
        for batch in range(num_batch_gen):
            noise = gen_noise(batch_size,(IMAGE_WIDTH,IMAGE_HEIGHT))
            (x,_) = next(gen)
            res = self.gan.generator.predict([noise,x],batch_size)
            for i in range(x.shape[0]):
                yuv = np.concatenate((x[i],res[i]),axis=2)
                rgb = LAB2RGB(yuv)
                fakepath = os.path.join(fake,f"{imnumber}.png")
                scipy.misc.imsave(fakepath,rgb)
                imnumber = imnumber + 1



    def train(self,batch_size = 16,epochs = 100,sample_interval=1,files=r'C:\lin\train\flower',store="models/"):
        #(x_train,y_train),(x_test,y_test) = self.get_data()
        #num_examples = x_train.shape[0]
        #num_batches = int(num_examples/ float(batch_size))
        #half_batch = int(batch_size / 2)
        #IMAGE_WIDTH = x_train.shape[1]
        #IMAGE_HEIGHT = x_train.shape[2]

        #NEW 256x256 IMAGE LOADER
        filenames = get_all_files(files,'.jpg')
        #filenames = filenames[:2000]
        num_examples = len(filenames)
        num_batches = int(num_examples/float(batch_size))
        half_batch = int(batch_size/2)
        gen = image_generator(filenames,half_batch)
        IMAGE_WIDTH = 256
        IMAGE_HEIGHT = 256
        GAN_FLAG = True
        

        #Adversarial ground truths
        #label_real = np.random.uniform(0.0,0.1,batch_size)#np.zeros((batch_size,1))
        #label_fake = np.random.uniform(0.9,1.0,batch_size)#np.ones((batch_size,1))
        #label_real = np.zeros((batch_size,1))
        #label_fake = np.ones((batch_size,1))

        labels_real = np.full((half_batch,1),1)
        labels_fake = np.full((half_batch,1),0)
        labels = np.concatenate((labels_fake,labels_real))

        for epoch in range(epochs):
            start = time.time()
            d_losses = []
            g_losses = []
            steps = 7
            for batch in range(num_batches):
                #select random batch of input images and generate associated noise
                #idx = np.random.randint(0,x_train.shape[0],batch_size)
                #x = x_train[idx]
                #real_imgs = y_train[idx]
                (fake_x,fake_y) = next(gen)
                (real_x,real_y) = next(gen)
                noise = gen_noise(half_batch,(IMAGE_WIDTH,IMAGE_HEIGHT))
                noise2 = gen_noise(half_batch,(IMAGE_WIDTH,IMAGE_HEIGHT))
                #generate fake images
                fake_imgs = self.gan.generator.predict([noise,fake_x])

                all_imgs = np.concatenate((fake_imgs,real_y))
                all_x = np.concatenate((fake_x,real_x))
                p = np.random.permutation(len(all_imgs))
                all_imgs = all_imgs[p]
                labels = labels[p]
                all_x = all_x[p]
                
                #test_img = np.concatenate((all_x[0],all_imgs[0]),axis=2)
                #test_img = LAB2RGB(test_img)
                #plt.figure()
                #plt.title("TRAIN")
                #plt.imshow(test_img)
                #plt.show()
                #plt.close()
                #train discriminator
                #if (epoch % 2) == 0:
                d_loss = np.array([])
                for k in range(steps):
                    (k_tx,k_ty) = next(gen)
                    (k_fx,k_fy) = next(gen)
                    k_n = gen_noise(half_batch,(IMAGE_WIDTH,IMAGE_HEIGHT))
                    gen_y = self.gan.generator.predict([noise,k_fx]) 
                    d_loss_real = self.gan.discriminator.train_on_batch([k_ty,k_tx],labels_real)
                    d_loss_fake = self.gan.discriminator.train_on_batch([gen_y,k_fx],labels_fake)
                    d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
                #d_loss = self.gan.discriminator.train_on_batch([all_imgs,all_x],labels)

                #train generator
                g_loss = self.gan.gan.train_on_batch([noise,fake_x],labels_real)
                end = time.time()
                d_losses.append(d_loss[0])
                g_losses.append(g_loss)
                text = "Epoch:%d [%d/%d] [D loss: %10.7f, acc.: %.2f%%] [G loss: %f] - %d     --- %f" % (epoch,batch,num_batches,np.mean(d_losses), 100*d_loss[1],np.mean(g_losses),end-start,d_loss[0])
                print (text,end='\r',flush=True)
                #keep track of losses and metrics
                #self.d_loss_a.append(d_loss[0])
                #self.gan_loss_a.append(g_loss)
                if (epoch % sample_interval == 0) and (batch == 0):
                    dpath = os.path.join(store,f"d_{epoch}.h5")
                    gpath = os.path.join(store,f"g_{epoch}.h5")
                    ganpath = os.path.join(store,f"gan_{epoch}.h5")
                    self.gan.discriminator.save(dpath)
                    self.gan.generator.save(gpath)
                    self.gan.gan.save(ganpath)
                    ridx = np.random.randint(0,fake_x.shape[0],1)
                    rand = fake_x[ridx[0]]
                    self.sample_images(self.gan,epoch,rand,text,True)
                    #self.plot_metrics(epoch)
            print()


    def plot_metrics(self,epoch):
        fig,ax = plt.subplots()
        ax.plot(self.d_loss_a,label="Discriminator loss")
        ax.plot(self.gan_loss_a,label="Generator loss")
        fig.legend()
        ax.set_title(f"Epoch {epoch}")
        fig.savefig(f"plots/{epoch}.png")
        plt.close()

    def sample_images(self,gan,epoch,rand,text,telegram = True,phase=''):
        #noise = gen_noise(1,shape=(rand.shape[0],rand.shape[1]))
        #imgs = gan.generator.predict([noise,np.expand_dims(rand,axis=0)])
        #yuv = np.concatenate((rand,imgs[0]),axis=2)
        #rgb = LAB2RGB(yuv)
        #scipy.misc.imsave(f"images/{phase}_{epoch}.png",rgb)
        #photo = open(f"images/{phase}_{epoch}.png", 'rb')
        #send_photo(rgb)
        send_message(text)
        

t = Trainer()
t.pretrain_generator()
t.pretrain_discriminator()
t.train()
