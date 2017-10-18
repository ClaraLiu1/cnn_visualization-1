#-*-coding:utf8-*-

import caffe
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

def vis_tensor(data):


	img = np.rollaxis(data,0,3)[:,:,0:3]
	b,g,r = cv2.split(img)
	img = cv2.merge([r,g,b])

	return  img
	# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	# cv2.imshow('image',img)
	# cv2.waitKey()

net_file='../alexnet/deploy.prototxt'
caffe_model='../alexnet/models/alexnet_iter_400.caffemodel'
mean_file='../data/mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
print(net.blobs['data'].data.shape)
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data',1) 
transformer.set_channel_swap('data', (2,1,0))
im = cv2.imread('473.jpg')
im = cv2.resize(im,(227,227))
ims = np.asarray(im)
print(ims.shape)
net.blobs['data'].data[...] = transformer.preprocess('data',ims)
out = net.forward()
print(net.blobs['data'].data.shape)

data1 = net.blobs['conv1'].data[0]
data2 = net.blobs['norm1'].data[0]
data3 = net.blobs['pool1'].data[0]
data4 = net.blobs['conv2'].data[0]
data5 = net.blobs['norm2'].data
data6 = net.blobs['pool2'].data[0]
data7 = net.blobs['conv3'].data[0]
data8 = net.blobs['conv4'].data[0]
data9 = net.blobs['conv5'].data[0]
data10 = net.blobs['pool5'].data[0]
prob = net.blobs['prob'].data[0]
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
print(data8.shape)
print(data9.shape)
print(data10.shape)

for layer_name, param in net.params.items():
	print (layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
'''1print(net.blobs)
OrderedDict([('data', <caffe._caffe.Blob object at 0x7f3087e43450>), ('conv1', <caffe._caffe.Blob object at 0x7f3087e43500>), 
('norm1', <caffe._caffe.Blob object at 0x7f3087e43558>), ('pool1', <caffe._caffe.Blob object at 0x7f3087e435b0>),
('conv2', <caffe._caffe.Blob object at 0x7f3087e43608>), ('norm2', <caffe._caffe.Blob object at 0x7f3087e43660>), 
('pool2', <caffe._caffe.Blob object at 0x7f3087e436b8>), ('conv3', <caffe._caffe.Blob object at 0x7f3087e43710>), 
('conv4', <caffe._caffe.Blob object at 0x7f3087e43768>), ('conv5', <caffe._caffe.Blob object at 0x7f3087e437c0>),
('pool5', <caffe._caffe.Blob object at 0x7f3087e43818>), ('fc6', <caffe._caffe.Blob object at 0x7f3087e43870>),
('fc7', <caffe._caffe.Blob object at 0x7f3087e438c8>), ('fc8', <caffe._caffe.Blob object at 0x7f3087e43920>),
('prob', <caffe._caffe.Blob object at 0x7f3087e43978>)])

print(net.params)
OrderedDict([('conv1', <caffe._caffe.BlobVec object at 0x7f5de4d44490>), 
('conv2', <caffe._caffe.BlobVec object at 0x7f5de4d44760>), 
('conv3', <caffe._caffe.BlobVec object at 0x7f5de4d446c0>), 
('conv4', <caffe._caffe.BlobVec object at 0x7f5de4d44710>), 
('conv5', <caffe._caffe.BlobVec object at 0x7f5de4d44620>), 
('fc6', <caffe._caffe.BlobVec object at 0x7f5de4d44800>),
('fc7', <caffe._caffe.BlobVec object at 0x7f5de4d448f0>),
('fc8', <caffe._caffe.BlobVec object at 0x7f5de4d44990>)])

for layer_name, param in net.params.items():
	print (layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
conv1	(96, 3, 11, 11) (96,)
conv2	(256, 48, 5, 5) (256,)
conv3	(384, 256, 3, 3) (384,)
conv4	(384, 192, 3, 3) (384,)
conv5	(256, 192, 3, 3) (256,)
fc6	(4096, 9216) (4096,)
fc7	(4096, 4096) (4096,)
fc8	(5, 4096) (5,)
'''
# img1 = vis_tensor(data1)
# img2 = vis_tensor(data2)
# img3 = vis_tensor(data3)
# img4 = vis_tensor(data4)
# img5 = vis_tensor(data5)
# img6 = vis_tensor(data6)
# img7 = vis_tensor(data7)
# img8 = vis_tensor(data8)
# plt.subplot(811),plt.imshow(img1)
# plt.subplot(812),plt.imshow(img2)
# plt.subplot(813),plt.imshow(img3)
# plt.subplot(814),plt.imshow(img4)
# plt.subplot(815),plt.imshow(img5)
# plt.subplot(816),plt.imshow(img6)
# plt.subplot(817),plt.imshow(img7)
# plt.subplot(818),plt.imshow(img8)
# plt.show()





