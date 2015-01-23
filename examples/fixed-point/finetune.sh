#../../build/tools/caffe train -solver solver.prototxt -weights ../../models/vgg/VGG_ILSVRC_16_layers_full_conv_4_by_4_link_conv2_1.caffemodel -gpu 0
../../build/tools/caffe train -solver solver_late.prototxt -weights ../../models/vgg/VGG_ILSVRC_16_layers_full_conv_4_by_4_link_last.caffemodel -gpu 0
