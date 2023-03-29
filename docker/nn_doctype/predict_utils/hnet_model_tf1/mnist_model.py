'''MNIST-rot model'''

# import os
import sys
# import time
sys.path.append('../')

# import numpy as np
import tensorflow as tf

import predict_utils.hnet_model_tf1.harmonic_network_lite as hn_lite
#from harmonic_network_helpers import *


def deep_mnist(args, x, train_phase):
   """The MNIST-rot model similar to the one in Cohen & Welling, 2016"""
   # Sure layers weight & bias
   order = 1
   # Number of Filters
   nf = int(args.n_filters)
   nf2 = int((args.filter_gain) * nf)
   nf3 = int((args.filter_gain ** 2) * nf)
   nf4 = int((args.filter_gain ** 3) * nf)
   bs = args.batch_size
   fs = args.filter_size
   ncl = args.n_classes
   sm = args.std_mult
   nr = args.n_rings

   # Create bias for final layer
   # bias = tf.compat.v1.get_variable('b7', shape=[args.n_classes],
   #                   initializer=tf.constant_initializer(1e-2))
   x = tf.reshape(x, shape=[bs,args.dim,args.dim,1,1,1])

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = hn_lite.conv2d(x, nf, 3, padding='SAME', n_rings=nr, name='1')
      cv1 = hn_lite.non_linearity(cv1, tf.nn.relu, name='1')

      cv2 = hn_lite.conv2d(cv1, nf, fs, padding='SAME', n_rings=nr, name='2')
      cv2 = hn_lite.batch_norm(cv2, train_phase, name='bn1')
      print(cv2)
   with tf.name_scope('block2') as scope:
      cv2 = hn_lite.mean_pool(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
      cv3 = hn_lite.conv2d(cv2, nf2, 3, padding='SAME', n_rings=nr, name='3')
      cv3 = hn_lite.non_linearity(cv3, tf.nn.relu, name='3')

      cv4 = hn_lite.conv2d(cv3, nf2, fs, padding='SAME', n_rings=nr, name='4')
      cv4 = hn_lite.batch_norm(cv4, train_phase, name='bn2')
      print(cv4)
   with tf.name_scope('block3') as scope:
      cv4 = hn_lite.mean_pool(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
      cv5 = hn_lite.conv2d(cv4, nf3, 3, padding='SAME', n_rings=nr, name='5')
      cv5 = hn_lite.non_linearity(cv5, tf.nn.relu, name='5')

      cv6 = hn_lite.conv2d(cv5, nf3, fs, padding='SAME', n_rings=nr, name='6')
      cv = hn_lite.batch_norm(cv6, train_phase, name='bn3')
      print(cv)

   with tf.name_scope('block4') as scope:
      cv = hn_lite.mean_pool(cv, ksize=(1,2,2,1), strides=(1,2,2,1))
      cv = hn_lite.conv2d(cv, nf4, 3, padding='SAME', n_rings=nr, name='77')
      cv = hn_lite.non_linearity(cv, tf.nn.relu, name='77')

      cv = hn_lite.conv2d(cv, nf4, fs, padding='SAME', n_rings=nr, name='8')
      cv = hn_lite.batch_norm(cv, train_phase, name='bn4')
      print(cv)

   # with tf.name_scope('block5') as scope:
   #    cv = hn_lite.mean_pool(cv, ksize=(1,2,2,1), strides=(1,2,2,1))
   #    cv = hn_lite.conv2d(cv, nf4, 2, padding='SAME', n_rings=nr, name='9')
   #    cv = hn_lite.non_linearity(cv, tf.nn.relu, name='9')
   #    cv = hn_lite.mean_pool(cv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))
   #    print(cv)
   #
   # with tf.name_scope('block6') as scope:
   #    cv = hn_lite.conv2d(cv, nf4, 2, padding='SAME', n_rings=nr, name='10')
   #    cv = hn_lite.non_linearity(cv, tf.nn.relu, name='10')
   #    cv = hn_lite.mean_pool(cv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))
   #    print(cv)

   # Final Layer
   with tf.name_scope('block10') as scope:
      # cv7 = hn_lite.conv2d(cv, ncl, fs, padding='SAME', n_rings=nr, phase=False,
      #          name='70')
      real = hn_lite.sum_magnitudes(cv)
      cv = tf.reduce_mean(real, axis=[1,2,3,4])
      # r = tf.compat.v1.layers.dense(cv, ncl*20)
      print(cv)
      r = tf.nn.tanh(cv)
      r = tf.compat.v1.layers.dense(r, ncl)
      # r = tf.nn.bias_add(r, bias)
      # r = tf.nn.sigmoid(r)  # activation
      # return
      return r