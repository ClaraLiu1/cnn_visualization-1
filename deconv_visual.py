#-*-coding-*-

__author__ = 'buyizhiyou'
__datat__ = '2017-10-17'
'''
利用反卷积可视
paper:Visualizing and Understanding Convolutional Networks  
author:Matthew D Zeiler, Rob Fergus
'''

import numpy as np
import sys
import random
import os
from os.path import isfile, join
import caffe
from os import listdir
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from scipy.stats import pearsonr, spearmanr
import cv2
import pdb
from reverse import  unpooling_layer,deconv_layer,derelu_layer


net_file='../alexnet/deploy.prototxt'
caffe_model='../alexnet/models/alexnet_iter_400.caffemodel'
mean_file='../data/mean.npy'

net = caffe.Classifier(net_file, caffe_model,  
                       mean=np.load(mean_file).mean(1).mean(1),  
                       channel_swap=(2,1,0),  
                       raw_scale=1,  
                       image_dims=(227, 227))  


def get_pool_indice(net,layer_number, no_feature_map):
    pool_layer = 'pool'+str(layer_number)
    conv_layer = 'conv'+str(layer_number)
    pool_indice= np.zeros(net.blobs[pool_layer].data.shape, dtype='int')
    #print (pool_indice.shape) #(1,96, 27, 27)
    #print (net.blobs[conv_layer].data[0].shape) #( 96, 55, 55)
    for i in range(0,net.blobs[conv_layer].data[0].shape[1]-1,2):
        for j in range(0,net.blobs[conv_layer].data[0].shape[2]-1,2):
            temp = net.blobs[conv_layer].data[0][:,i:i+3,j:j+3]
            #print (temp.shape)
            if temp.shape[1]!=3:
                    temp = np.append(temp,np.zeros((temp.shape[0],3-temp.shape[1],temp.shape[2])),axis=1)
            if temp.shape[2]!=3:
                    temp = np.append(temp,np.zeros((temp.shape[0],temp.shape[1],3-temp.shape[2])),axis=2)
            pool_indice[0,:,i//2,j//2]=np.argmax(np.reshape(temp,(temp.shape[0],9)),axis=1)
            for k in range(net.blobs[conv_layer].data.shape[1]):
                if(np.max(temp[k,:,:])==0):
                    pool_indice[0,k,i//2,j//2]=-1
    #print pool_indice[0,:,:,:].shape, net.blobs[conv_layer].data[4,:,:,:].shape
    return pool_indice[0,:,:,:]

def net_extract(net,image_name):  

    small_size = 256
    net.predict([caffe.io.load_image(image_name)])

def recon_conv5(net, no_feature_map, pool1_indice, pool2_indice):
    #pdb.set_trace()
    inputs = net.blobs['conv5'].data[0][no_feature_map:no_feature_map+1,:,:]
    temp = np.squeeze(net.blobs['conv5'].data[0][no_feature_map:no_feature_map+1,:,:])
    print ('conv5 blob shape:', temp.shape) # (13, 13)
    index = np.argmax(temp)
    x = index//temp.shape[0]
    y = index % temp.shape[0] #x,y(11, 3)
    print ('max index:',(x,y),np.max(temp), temp[x,y])
    filters = net.params['conv5'][0].data[no_feature_map:no_feature_map+1,:,:,:]
    #print(net.params['conv5'][1].data.shape) (256,)
    bias = net.params['conv5'][1].data[no_feature_map:no_feature_map+1]
    group=int(no_feature_map/(net.params['conv4'][0].data.shape[0]/2))
    print ('group:%d' % group)
    print ('conv5 filters and bias:', filters.shape, bias.shape)
    inputs = np.asarray([[[temp[x,y]]]]) #选取最大激活值

    print('inputs:',inputs,inputs.shape)
    
    outputs = deconv_layer(inputs, filters, bias)
    outputs = derelu_layer(outputs)
    temp_shape = net.blobs['conv4'].data.shape
    if(x-1<0):
        outputs = outputs[:,1-x:,:]
    if(x+1>temp_shape[2]-1):
        outputs = outputs[:,:temp_shape[2]-1-x-1,:]
    if(y-1<0):
        outputs = outputs[:,:,1-y:]
    if(y+1>temp_shape[3]-1):
        outputs = outputs[:,:,:temp_shape[3]-1-y-1]
    boundary = [max(x-1,0), min(x+1, temp_shape[2]-1), max(y-1,0), min(y+1, temp_shape[3]-1)]    
    print ('reconstructed conv4 blob shape and boundary:', outputs.shape, boundary)
    recon_conv4 = outputs
    boundary0 = boundary
    
    #pdb.set_trace()
    filters = net.params['conv4'][0].data[group*192:(group+1)*192,:,:]
    bias = net.params['conv4'][1].data[group*192:(group+1)*192].reshape(192,1,1)
    print ('conv4 filters and bias:', filters.shape, bias.shape)
    
    outputs = deconv_layer(recon_conv4, filters, bias)
    outputs = derelu_layer(outputs)
    temp_shape = net.blobs['conv3'].data.shape
    if(boundary0[0]-1<0):
        outputs = outputs[:,1-boundary0[0]:,:]
    if(boundary0[1]+1>temp_shape[2]-1):
        outputs = outputs[:,:temp_shape[2]-1-boundary0[1]-1,:]
    if(boundary0[2]-1<0):
        outputs = outputs[:,:,1-boundary0[2]:]
    if(boundary0[3]+1>temp_shape[3]-1):
        outputs = outputs[:,:,:temp_shape[3]-1-boundary0[3]-1]
    boundary = [max(boundary0[0]-1,0), min(boundary0[1]+1, temp_shape[2]-1), max(boundary0[2]-1,0), min(boundary0[3]+1, temp_shape[3]-1)]
 
    recon_conv3 = outputs
    boundary0 = boundary
    
    #pdb.set_trace()
    filters = net.params['conv3'][0].data[group*192:(group+1)*192,:,:,:]
    bias = net.params['conv3'][1].data[group*192:(group+1)*192].reshape(192,1,1)
    print ('conv3 filters and bias:', filters.shape, bias.shape)
    
    outputs = deconv_layer(recon_conv3, filters, bias)
    outputs = derelu_layer(outputs)
    
    #pdb.set_trace()
    temp_shape = net.blobs['pool2'].data.shape#(1, 256, 13, 13)
    if(boundary0[0]-1<0):
        outputs = outputs[:,1-boundary0[0]:,:]
    if(boundary0[1]+1>temp_shape[2]-1):
        outputs = outputs[:,:temp_shape[2]-1-boundary0[1]-1,:]
    if(boundary0[2]-1<0):
        outputs = outputs[:,:,1-boundary0[2]:]
    if(boundary0[3]+1>temp_shape[3]-1):
        outputs = outputs[:,:,:temp_shape[3]-1-boundary0[3]-1]
    boundary = [max(boundary0[0]-1,0), min(boundary0[1]+1, temp_shape[2]-1), max(boundary0[2]-1,0), min(boundary0[3]+1, temp_shape[3]-1)]
    #print 'pool2 blob shape:', outputs.shape, boundary 
    pool2_index = pool2_indice[:, boundary[0]:boundary[1]+1, boundary[2]:boundary[3]+1]
    conv2_shape = [pool2_index.shape[0], pool2_index.shape[1]*2+1, pool2_index.shape[2]*2+1]
    print ('reconstructed pool2 shape and boundary:', outputs.shape, boundary) # (256, 5, 7) [8, 12, 0, 6]
    
    recon_conv2 = unpooling_layer(outputs, pool2_index,conv2_shape)
    temp_shape = net.blobs['conv2'].data.shape
    boundary1 = [boundary[0]*2, min((boundary[1])*2+2, temp_shape[2]-1), boundary[2]*2, min(boundary[3]*2+2, temp_shape[3]-1)]
    print ('reconstructed conv2 shape and boundary:', recon_conv2.shape, boundary1)

    #pdb.set_trace()
    conv2_part1 = recon_conv2[:int(recon_conv2.shape[0]/2),:,:]
    conv2_part2 = recon_conv2[int(recon_conv2.shape[0]/2):,:,:]
    filters = net.params['conv2'][0].data[:,:,:,:]
    bias = net.params['conv2'][1].data.reshape(-1,1,1)
    print ('conv2 filters and bias:',filters.shape, bias.shape)
    filters_part1 = filters[:int(recon_conv2.shape[0]/2),:,:,:]
    filters_part2 = filters[int(recon_conv2.shape[0]/2):,:,:,:]
    bias_part1 = bias[:int(recon_conv2.shape[0]/2)]
    bias_part2 = bias[int(recon_conv2.shape[0]/2):]
    recon_pool1_part1 = deconv_layer(conv2_part1, filters_part1, bias_part1)
    recon_pool1_part2 = deconv_layer(conv2_part2, filters_part2, bias_part2)
    
    recon_pool1 = np.concatenate((recon_pool1_part1, recon_pool1_part2))
    recon_pool1 = derelu_layer(recon_pool1)
   
    #pdb.set_trace()
    temp_shape = net.blobs['pool1'].data.shape
    #print (temp_shape, recon_pool1.shape)#(1, 96, 27, 27) (96, 15, 19)
    if(boundary1[0]-2<0):
        recon_pool1 = recon_pool1[:,2-boundary1[0]:,:]
    if(boundary1[1]+2>temp_shape[2]-1):
        recon_pool1 = recon_pool1[:,:temp_shape[2]-1-boundary1[1]-2,:]
    if(boundary1[2]-2<0):
        recon_pool1 = recon_pool1[:,:,2-boundary1[2]:]
    if(boundary1[3]+2>temp_shape[3]-1):
        recon_pool1 = recon_pool1[:,:,:temp_shape[3]-1-boundary1[3]-2]
    boundary2 = [max(boundary1[0]-2,0), min(boundary1[1]+2, temp_shape[2]-1), max(boundary1[2]-2,0), min(boundary1[3]+2, temp_shape[3]-1)]
    print ('reconstructed pool1 shape and boundary:', recon_pool1.shape,boundary2)# (96, 13, 17) [14, 26, 0, 16]
    pool1_index = pool1_indice[:, boundary2[0]:boundary2[1]+1, boundary2[2]:boundary2[3]+1]
    conv1_shape = [pool1_index.shape[0], pool1_index.shape[1]*2+1, pool1_index.shape[2]*2+1]
    recon_conv1 = unpooling_layer(recon_pool1, pool1_index,conv1_shape)
    temp_shape = net.blobs['conv1'].data.shape
    boundary3 = [boundary2[0]*2, min((boundary2[1])*2+2, temp_shape[2]-1), boundary2[2]*2, min(boundary2[3]*2+2, temp_shape[3]-1)]
    print ('reconstructed conv1 shape and boundary:', recon_conv1.shape, boundary3)# (96, 27, 35) [28, 54, 0, 34]
    
    filters = net.params['conv1'][0].data[:,:,:,:]
    bias = net.params['conv1'][1].data.reshape(-1,1,1)
    print ('conv1 filters and bias:', filters.shape, bias.shape)#conv1 filters and bias: (96, 3, 11, 11) (96, 1, 1)
    
    #show reconstructed data
    recon_data = deconv_layer(recon_conv1, filters, bias,method='same')
    recon_data= np.transpose(np.squeeze(recon_data),(1,2,0))
    recon_data= (recon_data-np.min(recon_data))/(np.max(recon_data)-np.min(recon_data))
    
    print ('reconstructed data shape', recon_data.shape)#reconstructed data shape (27, 35, 3)


    temp_shape = net.blobs['data'].data.shape
    #print (temp_shape)
    print ('reconstructed data shape and boundary:', recon_data.shape, boundary3)
    return boundary3, recon_data


conv5 = net.blobs['conv5'].data
p_ = net.blobs['prob'].data
max_conv5 = np.zeros((conv5.shape[0],conv5.shape[1]))
mean_conv5 = np.zeros((conv5.shape[0],conv5.shape[1]))
#print(conv5.shape)
#print(p_.shape)
for i in range(conv5.shape[0]):
    for j in range(conv5.shape[1]):
        max_conv5[i,j] = np.max(conv5[i,j,:,:])
        mean_conv5[i,j] = np.sum(conv5[i,j,:,:])/169

file = '512.jpg'
no_feature_map = 150
net_extract(net,file)
# img = np.transpose(net.blobs['data'].data[0],(1,2,0))
# b,g,r = cv2.split(img)
# img = cv2.merge([r,g,b])
img = cv2.imread(file)
img = img[:,:,::-1]
plt.subplot(1,3,1)
plt.imshow(img)
pool1_indice = get_pool_indice(net,1,no_feature_map)
pool2_indice = get_pool_indice(net,2,no_feature_map)
plt.subplot(1,3,2)
#print (pool1_indice.shape)#(96, 27, 27)
#print(pool2_indice.shape)#(256, 13, 13)
boundary, outputs_layer1 = recon_conv5(net, no_feature_map, pool1_indice,pool2_indice)
plt.imshow(outputs_layer1[:,:,::-1])
plt.subplot(1,3,3)
xmin = max(0, boundary[0]*4);xmax = boundary[1]*4
ymin = max(0,boundary[2]*4);ymax = boundary[3]*4
#print ((xmin, xmax,ymin,ymax))
plt.imshow(img[xmin:xmax,ymin:ymax,:])
plt.show()