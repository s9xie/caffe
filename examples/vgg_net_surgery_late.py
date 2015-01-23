# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import matplotlib.pyplot as plt

# Load the original network and extract the fully-connected layers' parameters.
net = caffe.Net('../models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', '../models/vgg/VGG_ILSVRC_16_layers.caffemodel')
params_full = ['fc6', 'fc7', 'fc8']
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_full}
for fc in params_full:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

net_full_conv = caffe.Net('../examples/fixed-point/deploy_late.prototxt', '../models/vgg/VGG_ILSVRC_16_layers.caffemodel')
params_full_conv = ['context_conv1', 'cccp1', 'conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'fc6-conv', 'fc7-conv', 'voc-fc8-conv']
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

params_fc_conv = ['fc7-conv']
params = ['fc7']
#fc6 subsampling special case, fc8 starts from scratch
# conv_params = {name: (weights, biases)}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_fc_conv):
    conv_params[pr_conv][1][...] = fc_params[pr][1]

for pr, pr_conv in zip(params, params_fc_conv):
    out, in_, h, w = conv_params[pr_conv][0].shape
    W = fc_params[pr][0].reshape((out, in_, h, w))
    conv_params[pr_conv][0][...] = W

#Special case for fc6 and fc6-conv
out, in_, h, w = conv_params['fc6-conv'][0].shape
print out, in_, h, w

shrinked_filter = np.zeros([out, in_, h, w])
W = fc_params['fc6'][0].reshape((out, in_, 7, 7))
for i in range(0, out):
	for j in range(0, in_):
		shrinked_filter[i][j][...] = W[i][j][::2,::2]

conv_params['fc6-conv'][0][...] = shrinked_filter

#initialization different
#conv_params['aug_conv2_1'][0][...] = np.concatenate((fc_params['conv2_1'][0][...], np.random.normal(0, 0.05, [128,64,3,3])), axis=1)
#?Correct? part correspondence

net_full_conv.save('../models/vgg/VGG_ILSVRC_16_layers_full_conv_4_by_4_link_last.caffemodel')


for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)



#%matplotlib inline

# load input and configure preprocessing
#im = caffe.io.load_image('images/cat.jpg')
#net_full_conv.set_phase_test()
#net_full_conv.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'))
#net_full_conv.set_channel_swap('data', (2,1,0))
#net_full_conv.set_raw_scale('data', 255.0)
## make classification map by forward and print prediction indices at each location
#out = net_full_conv.forward_all(data=np.asarray([net_full_conv.preprocess('data', im)]))
#print out['prob'][0].argmax(axis=0)
