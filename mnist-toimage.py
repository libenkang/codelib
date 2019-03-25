# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib
import os
#from PIL import Image
import scipy.misc

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
    
for i in range(20):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28,28)
    filename = save_dir + 'mnist_train_%d.jpg' %i
#    matplotlib.image.imsave(filename,image_array)
    #im = Image.fromarray(image_array)
    #im = im.convert('RGB')
    #im.save(filename)
    im = scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0)
    im.save(filename)
