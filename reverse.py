#-*-coding:utf8-*-

__author__ = 'buyizhiyou'
__data__ = '2017-10-18'

'''
 反卷积，反池化，反激活算子
'''

import numpy as np
import sys
import random
import os
from os.path import isfile, join
from scipy.signal import convolve2d
import cv2
import pdb



def unpooling_layer(inputs, indice, outputs_shape):
    outputs = np.zeros(outputs_shape)  #初始化
    for i in range(indice.shape[1]):
        for j in range(indice.shape[2]):
            for k in range(indice.shape[0]):
                index = indice[k,i,j]
                if(index == -1):
                    continue
                x_offset = int(index/3);y_offset = index-x_offset*3
                outputs[k,i*2+x_offset,y_offset+j*2] = inputs[k,i,j] #最大值位置赋值
    return outputs

def deconv_layer(inputs,filters,bias,method = 'full'):
    #pdb.set_trace()
    filters = np.transpose(filters,(1,0,2,3))
    #print 'deconv filter shape:', filters.shape
    #filters = filters[:,:,::-1,::-1]
    filter_shape = filters.shape
    outputs = np.zeros((filter_shape[0],inputs.shape[1]+filters.shape[2]-1,inputs.shape[2]+filters.shape[3]-1))
    if method=='full':
        outputs = np.zeros((filter_shape[0],inputs.shape[1]+filters.shape[2]-1,inputs.shape[2]+filters.shape[3]-1))
    else:
        outputs = np.zeros((filter_shape[0],inputs.shape[1],inputs.shape[2]))
    #print inputs.shape, outputs.shape
    intputs = inputs-bias
    for i in range(filter_shape[0]):
        for j in range(filter_shape[1]):
            outputs[i,:,:] += convolve2d(inputs[j,:,:], filters[i,j,:,:],method)
    return outputs

def derelu_layer(inputs):
    outputs = np.maximum(inputs,np.zeros(inputs.shape))#relu
    return outputs

