import numpy as np
import os
import random 
import sys
import csv
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import losses
from keras import optimizers
from keras.utils import plot_model

from utils import io
from model import model_build

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 500
isValid = 1
#zoom_range = 0.2
zoom_range = 0.24

def gen_valid_set(feats, labels, frac):
    length = len(feats)
    points = int(length * frac)
    pairs = list(zip(feats, labels))
    random.shuffle(pairs)

    feats, labels = zip(*pairs)
    feats = np.array(feats)
    labels = np.array(labels)
    return feats[:(length - points)], labels[:(length - points)], \
           feats[(length - points):], labels[(length - points):]


def testing(X, model, path):
    prd_class = model.predict_classes(X)
    print(prd_class)
    with open("predict/{}.csv".format(path), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(X)):
            csv_writer.writerow([i]+[prd_class[i]])

def img_flip(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = np.flip(img, axis=1)
    return imgs

def main():
    tr_feats, tr_labels = io.read_dataset()
    tr_feats_flip = img_flip(tr_feats)
    tr_feats = np.concatenate((tr_feats, tr_feats_flip), axis=0)
    tr_labels = np.concatenate((tr_labels, tr_labels), axis=0)
    print(len(tr_feats))
    te_feats = io.read_dataset('test', False)
    tr_feats, tr_labels, val_feats, val_labels = gen_valid_set(tr_feats, tr_labels, 0.1)
    print(np.shape(tr_feats))
    '''
    train_gen = ImageDataGenerator(rotation_range=25, 
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[1-zoom_range, 1+zoom_range],
                                    horizontal_flip=True)
                                    '''
    train_gen = ImageDataGenerator(rotation_range=30,
                                    width_shift_range=0.15,
                                    height_shift_range=0.06,
                                    shear_range=0.1,
                                    zoom_range=[1-zoom_range, 1+zoom_range])
    train_gen.fit(tr_feats)
    # build CNN model
    model = model_build(input_shape, num_classes)

    # model training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model_feature = "3_Dense_CNN"
    #filepath = "ckpt/weights_early_" + model_feature + ".h5"

    callbacks = []
    modelcheckpoint = ModelCheckpoint("ckpt/weights.{epoch:03d}-{val_acc:.5f}.h5", monitor='val_acc', save_best_only=True, mode='max')
    callbacks.append(modelcheckpoint)
    csv_logger = CSVLogger('log/cnn_log.csv', separator=',', append=False)
    callbacks.append(csv_logger)
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
    callbacks.append(es)
    model.fit_generator(train_gen.flow(tr_feats, tr_labels, batch_size=batch_size),
                        steps_per_epoch=1*tr_feats.shape[0]//batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_feats, val_labels))
    model.save("model/final_predict.h5")

    score = model.evaluate(val_feats, val_labels)
    print("\nValidation Acc for \"final_model\": {}\n".format(score[1]))

'''
    model = load_model(filepath)
    score = model.evaluate(val_feats, val_labels)
    print("\nValidation Acc for \"load_model\": {}\n".format(score[1]))
    te_feats = io.read_dataset('test', False)
    predict_path = model_feature + str(score[1])
    testing(te_feats, model, predict_path)
    '''

if( __name__ == '__main__'):
    main() 
