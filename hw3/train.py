import numpy as np
import os
import random 
import sys
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
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 150
isValid = 1
model_name = ""

def gen_valid_set(feats, labels, frac):
    length = len(feats)
    points = int(length * frac)
    #pairs = list(zip(feats, labels))
    #datas = random.shuffle(pairs)

    #feats, labels = zip(*datas)
    return feats[:(length - points)], labels[:(length - points)], \
           feats[(length - points):], labels[(length - points):]


def testing(X, model):
    global model_name
    prd_class = model.predict_classes(X)
    print(prd_class)
    with open("predict/{}.csv".format(model_name), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(X)):
            csv_writer.writerow([i]+[prd_class[i]])

def main():
    global model_name
    tr_feats, tr_labels = io.read_dataset()
    te_feats = io.read_dataset('test', False)
    tr_feats, tr_labels, val_feats, val_labels = gen_valid_set(tr_feats, tr_labels, 0.1)
    train_gen = ImageDataGenerator(rotation_range=25, 
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[1-zoom_range, 1+zoom_range],
                                    horizontal_flip=True)
    train_gen.fit(tr_feats)
    print(np.shape(tr_feats))
    # build CNN model
    model = model_build(input_shape, num_classes)

    # model training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    callbacks = []
    filepath = "ckpt/weights_early_new_CNN.h5"
    modelcheckpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
    callbacks.append(modelcheckpoint)
    csv_logger = CSVLogger('log/cnn_log.csv', separator=',', append=False)
    callbacks.append(csv_logger)
    es = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
    callbacks.append(es)
    model.fit_generator(train_gen.flow(tr_feats, tr_labels, batch_size=batch_size),
                        steps_per_epoch=tr_feats.shape[0]//batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_feats, val_labels))
    modle.save("final_predict.h5")

    te_feats = io.read_dataset('test', False)
    model.load_model(filepath)
    testing(te_feats, model)

if( __name__ == '__main__'):
    main() 
