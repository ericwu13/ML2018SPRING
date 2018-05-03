from keras.models import load_model, Input, Model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import average, concatenate
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from utils import io
import sys
import numpy as np
import csv
import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def img_flip(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = np.flip(img, axis=1)
    return imgs

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

def ensembleModels(models, name):
    y_model = []
    for i in models:
        y_model.append(load_model(i))

    model_input = Input(shape=y_model[0].input_shape[1:])

    yModels=[model(model_input) for model in y_model]
    print(yModels[0][-1])
    print(yModels[1][-1])
    print(yModels[2][-1])
    #yAvg = average(yModels)
    yCon = concatenate(yModels, axis=1)
    final = Dense(192, activation='tanh')(yCon)
    final = Dense(64, activation='tanh')(final)
    final = Dense(7, activation='softmax')(final)
    modelEns = Model(inputs=model_input, outputs=final, name='ensemble')
    modelEns.save(name+'.h5')
    return modelEns

tr_feats, tr_labels = io.read_dataset()

#mean, std = np.mean(tr_feats, axis=0), np.std(tr_feats, axis=0)
#np.save('attr.npy', [mean, std])
#tr_feats = (tr_feats - mean) / (std + 1e-20)

tr_feats, tr_labels, val_feats, val_labels = gen_valid_set(tr_feats, tr_labels, 0.1)
tr_feats_flip = img_flip(tr_feats)
tr_feats = np.concatenate((tr_feats, tr_feats_flip), axis=0)
tr_labels = np.concatenate((tr_labels, tr_labels), axis=0)
tr_feats, tr_labels, va, vas = gen_valid_set(tr_feats, tr_labels, 0)

te_feats = io.read_dataset('test', False)
num_model = int(sys.argv[1])
name = sys.argv[2]
models = []
for i in range(num_model):
    models.append(sys.argv[i+3])

model = ensembleModels(models, name)
model.summary()
model.get_layer('sequential_1').trainable = False
model.get_layer('sequential_2').trainable = False
model.get_layer('sequential_3').trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = []
modelcheckpoint = ModelCheckpoint("ckpt_voting_64/weights.{epoch:03d}-{val_acc:.5f}.h5", monitor='val_acc', save_best_only=True, mode='max', verbose=1)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('log/cnn_log_voting_64.csv', separator=',', append=False)
callbacks.append(csv_logger)
es = EarlyStopping(monitor='val_loss', patience=70, verbose=1, mode='min')
callbacks.append(es) 
model.fit(tr_feats, tr_labels, batch_size=128,
                    epochs=500,
                    callbacks=callbacks,
                    validation_data=(val_feats, val_labels))
y_prob = model.predict(te_feats)
y_classes = y_prob.argmax(axis=-1)

with open("predict/ensemble{}.csv".format(str(num_model) + "_" + name), 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(te_feats)):
        csv_writer.writerow([i]+[y_classes[i]])
