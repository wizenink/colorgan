import os
import numpy as np
from skimage.io import imshow,imread
from skimage.color import rgb2lab,lab2rgb
from skimage.util import img_as_float64
import matplotlib.pyplot as plt

def RGB2LAB(rgb):
    #img = rgb / 255.0
    img = img_as_float64(rgb)
    lab = rgb2lab(img)
    #lab = (lab + [0, 128, 128]) / [100.0, 255.0, 255.0]
    #lab = lab / [1.0,128.0,128.0]
    return lab
def LAB2RGB(lab):
    #lab = (lab * [100, 255, 255]) - [0, 128, 128]
    #lab = lab * [1.0,128.0,128.0]
    rgb = lab2rgb(lab)
    return rgb
    
def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    #yuv[:,:,1:]+=128.0
    return yuv

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    #rgb[:,:,0]-=179.45477266423404
    #rgb[:,:,1]+=135.45870971679688
    #rgb[:,:,2]-=226.8183044444304
    #return rgb.clip(min=0.0,max=255.0)
    return rgb


def channel_split(image):
    a = image[:,:,0]
    b = image[:,:,1]
    c = image[:,:,2]
    return a,b,c

def testyuv():
    currentdir = os.path.dirname(os.path.realpath(__file__))
    lena_path = os.path.join(currentdir,"testimages/lena.jpg")
    lena_rgb = imread(lena_path)[:,:,:3]
    lena_yuv = YUV2RGB(lena_rgb)

    r,g,b = channel_split(lena_rgb)
    y,u,v = channel_split(lena_yuv)

    f,axarr = plt.subplots(4,2)

    axarr[0,0].imshow(lena_rgb)
    axarr[0,0].set_axis_off()
    axarr[0,0].set_title("RGB")

    axarr[1,0].imshow(r,cmap='gray')
    axarr[1,0].set_axis_off()
    axarr[1,0].set_title("R")

    axarr[2,0].imshow(g,cmap='gray')
    axarr[2,0].set_axis_off()
    axarr[2,0].set_title("G")

    axarr[3,0].imshow(b,cmap='gray')
    axarr[3,0].set_axis_off()
    axarr[3,0].set_title("B")

#    axarr[0,1].imshow(lena_yuv/255.0)
    axarr[0,1].set_axis_off()
    axarr[0,1].set_title("YUV")

    axarr[1,1].imshow(y/255.0,cmap='gray')
    axarr[1,1].set_axis_off()
    axarr[1,1].set_title("Y")

    axarr[2,1].imshow(u/255.0,cmap='gray')
    axarr[2,1].set_axis_off()
    axarr[2,1].set_title("U")

    axarr[3,1].imshow(v/255.0,cmap='gray')
    axarr[3,1].set_axis_off()
    axarr[3,1].set_title("V")
    
    plt.show()
    return

if __name__ == '__main__':
    testyuv()