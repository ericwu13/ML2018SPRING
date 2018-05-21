import sys, argparse, os
import keras
import readline
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Sequential
from keras.layers import Dense, GRU, Flatten
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from rnn_model import RNN
from preprocess import *

train_path = 'data/training_label.txt'
test_path = 'data/testing_data.txt'
semi_path = 'data/training_nolabel.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment classification')
    parser.add_argument('model')
    parser.add_argument('action', choices=['train','semi', 'else'])
    parser.add_argument('--load', action='store_true')

    # training argument
    parser.add_argument('--gpu_fraction', default=0.3, type=float)
    parser.add_argument('--train_path', default=train_path, type=str)
    parser.add_argument('--semi_path', default=semi_path, type=str)

    # model parameter
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--w2vdim', default=256, type=int)
    parser.add_argument('--max_length', default=40, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--threshold', default=0.05,type=float)

    # output path for your prediction
    parser.add_argument('--result_path', default='result.csv',)

    # put model in the same directory
    parser.add_argument('--load_path', default = None)
    parser.add_argument('--save_dir', default = 'ckpt/')
    parser.add_argument('--rm_stop_words', action='store_true')
    return parser.parse_args()

def main(args):
    '==============================================='
    '========= Data Manager Initialziation ========='
    '==============================================='
    dm = DataManager()
    dm.add_rawData('train_data', args.train_path)
    dm.add_rawData('semi_data', args.semi_path, False)
    save_dir = "ckpt/{}/".format(args.model)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.load is True:   
        (X, Y), semi_all_X = preprocess_trainData(dm, mode='train', maxlen=args.max_length)
    else:                   
        (X, Y), semi_all_X = preprocess_trainData(dm, mode='retrain', dim=args.w2vdim, maxlen=args.max_length)

    '==============================================='
    '=========== Model Forming & Training =========='
    '==============================================='
    (X, Y), (X_val,Y_val) = dm.split_data([X,Y], args.val_ratio)
    model = RNN(args)
    model.summary()
    earlystopping = EarlyStopping(monitor='val_acc', patience = 15, verbose=1, mode='max')
    save_path = save_dir + "weights.{epoch:03d}-{val_acc:.5f}.h5"
    if args.action == 'train':
        checkpoint = ModelCheckpoint(filepath=save_path,\
                                     verbose=1, save_best_only=True,
                                     monitor='val_acc', mode='max' )
        history = model.fit(X, Y, validation_data=(X_val, Y_val),\
                        epochs=50, batch_size=512, callbacks=[checkpoint, earlystopping] )
    if args.action == 'semi':
        weight_path = save_dir+ "weight.h5"
        checkpoint = ModelCheckpoint(filepath=weight_path,\
                                     verbose=1, save_best_only=True,\
                                     monitor='val_acc', mode='max' )
        # repeat 10 times
        for i in range(10):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data(semi_all_X, semi_pred, args.threshold)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=512,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(weight_path):
                print ('load model from %s' % weight_path)
                model.load_weights(weight_path)
            model.save(save_dir+"model_forloading.h5")

if __name__ == '__main__':
    args = parse_args()
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session(args.gpu_fraction))
    main(args)
