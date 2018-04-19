
#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
# from marcos import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

def main(): 
    parser = argparse.ArgumentParser(prog='visualize_filter.py',
            description='ML-Assignment3 filter visualization.')
    parser.add_argument('--model', type=str, metavar='<#model>', required=True)
    parser.add_argument('--data', type=str, metavar='<#data>', required=True)
    parser.add_argument('--attr', type=str, metavar='<#attr>', required=True)
    parser.add_argument('--filter_dir', type=str, metavar='<#filter_dir>', default='./image/filter')
    args = parser.parse_args()

    data_name = args.data
    attr_name = args.attr
    model_name = args.model
    filter_dir = args.filter_dir

    print('load model')
    emotion_classifier = load_model(model_name)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)

    print('load data') 
    f = open(data_name, 'rb')
    f.seek(0)
    data = pickle.load(f)
    X, Y, _X = data[0], data[1], data[2]

    print('load attr')
    attr = np.load(attr_name)
    mean, std = attr[0], attr[1]

    input_img = emotion_classifier.input
    name_ls = ['conv2d_1', 'p_re_lu_1', 'conv2d_6', 'leaky_re_lu_6']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    choose_id = -4998
    for cnt, fn in enumerate(collect_layers):
        photo = X[choose_id].reshape(1, 48, 48, 1)
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        #nb_filter = im[0].shape[3]
        nb_filter = 32
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('filter {}'.format(i))
            plt.tight_layout()
        #fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join(filter_dir, 'vis')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path, '{}-{}'.format(name_ls[cnt], choose_id)))

if __name__ == '__main__':
    main()
