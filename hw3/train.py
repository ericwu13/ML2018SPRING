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
from model_final_good import model_build
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
'''

height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 500
isValid = 1
zoom_range = 0.2
#zoom_range = 0.24

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
    '==================== Data Forming ===================='

    tr_feats, tr_labels = io.read_dataset(path=True, data_path=sys.argv[1])

    #mean, std = np.mean(tr_feats, axis=0), np.std(tr_feats, axis=0)
    #np.save('attr.npy', [mean, std])
    #tr_feats = (tr_feats - mean) / (std + 1e-20)

    tr_feats, tr_labels, val_feats, val_labels = gen_valid_set(tr_feats, tr_labels, 0.1)
    tr_feats_flip = img_flip(tr_feats)
    tr_feats = np.concatenate((tr_feats, tr_feats_flip), axis=0)
    tr_labels = np.concatenate((tr_labels, tr_labels), axis=0)
    tr_feats, tr_labels, va, vas = gen_valid_set(tr_feats, tr_labels, 0)


    print(len(tr_feats))
    print(np.shape(tr_feats))
    train_gen = ImageDataGenerator(rotation_range=25, 
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[1-zoom_range, 1+zoom_range],
                                    horizontal_flip=True)
    train_gen.fit(tr_feats)

    '==================== Training Setting ==================='
    model = model_build(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_feature = "3_Dense_CNN"
    #filepath = "ckpt/weights_early_" + model_feature + ".h5"
    callbacks = []
    modelcheckpoint = ModelCheckpoint("ckpt_center_norm/weights.{epoch:03d}-{val_acc:.5f}.h5", monitor='val_acc', save_best_only=True, mode='max', verbose=1)
    callbacks.append(modelcheckpoint)
    csv_logger = CSVLogger('log/cnn_log_center_norm.csv', separator=',', append=False)
    callbacks.append(csv_logger)
    es = EarlyStopping(monitor='val_loss', patience=70, verbose=1, mode='min')
    callbacks.append(es)

    model.fit_generator(train_gen.flow(tr_feats, tr_labels, batch_size=batch_size),
                        steps_per_epoch=1*tr_feats.shape[0]//batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_feats, val_labels))

    model.save("model/final_predict.h5")
    score = model.evaluate(val_feats, val_labels)
    print("\nValidation Acc for \"final_model\": {}\n".format(score[1]))


if( __name__ == '__main__'):
    main() 
