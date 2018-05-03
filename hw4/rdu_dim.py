import argparse
from keras.models import load_model
import numpy as np
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
set_session(tf.Session(config=config))

def gen_valid_set(feats, frac):
    length = len(feats)
    points = int(length * frac)
    random.shuffle([feats])

    feats = np.array(feats)
    return feats[:(length - points)], feats[(length - points):]


def parse_args():
    parser = argparse.ArgumentParser(description='Dimension Reduction')
    parser.add_argument('--PCA', type=int, metavar='<#pca_components>')
    parser.add_argument('--encoder', type=int, metavar='<#encoder_components>')
    parser.add_argument('--reconstruct', type=int, metavar='<#reconstruct>')
    parser.add_argument('--read', type=str, metavar='<#read>')
    parser.add_argument('--data', type=str, metavar='<#data>')
    return parser.parse_args()

def auto_encoder(images, encoding_dim):
    x_train = images / 255.

    input_img = Input(shape=(28*28,))
    # "encoded" layer: use the input image to produce encoded_dim image
    encoded = Dense(128, activation='tanh')(input_img)
    encoded = Dense(64, activation='tanh')(encoded)
    encoded = Dense(encoding_dim, activation='tanh')(encoded)
    # "decoded" layer: use the encoded image to reconstruct original image
    decoded = Dense(64, activation='tanh')(encoded)
    decoded = Dense(128, activation='tanh')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    
    # map the input to the reconstruction image
    autoencoder = Model(input_img, decoded)
    # the model encode the "image"
    encoder = Model(input_img, encoded)
    
    # decoder model
    autoencoder.summary()
    adam = Adam(lr=5e-4)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')

    ckpt = ModelCheckpoint("ckpt_adam/weights.{epoch:03d}-{val_loss:.5f}.h5",\
                    monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
    autoencoder.fit(x_train, x_train,
                epochs=400,
                batch_size=128,
                shuffle=True,
                callbacks=[ckpt, es],
                validation_split=0.1
                )
    encoded_imgs = encoder.predict(x_train)
    #decoded_imgs = decoder.predict(encoded_imgs) * 255
    decoded_imgs = autoencoder.predict(x_train) * 255
    np.save('encoder_images/encoded_{}'.format(encoding_dim), encoded_imgs)

    recon_plot(images, decoded_imgs, 'encoder_{}'.format(encoding_dim))

def recon_plot(images, recon_imgs, title):
    n = 10
    plt.figure(figsize=(20, 4))
    plt.title(title)
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def main(args):
    images = np.load(args.data)
    input_shape = (28, 28)
    if args.PCA is not None:
        print("PCA dim reduced with components = \"" + str(args.PCA) + "\"")
        # state= 120 -> pca 350 0.99996
        # state= 110 -> pca 400 ??
        # state= 10 -> pca 400 ??
        pca = PCA(args.PCA, whiten=True, random_state=120)
        pca_images = pca.fit_transform(images)
        np.save('pca_images/pca_{}'.format(args.PCA), pca_images)
        components_ = pca.components_
        recon_images = np.dot(pca_images, components_)
        recon_plot(images, recon_images, 'PCA_{}'.format(args.PCA))
    if args.encoder is not None:
        print("Auto-encoder dim reduced with components = \"" + str(args.encoder) + "\"")
        auto_encoder(images, args.encoder)
    if args.read is not None:
        autoencoder = load_model(args.read)
        input_img = autoencoder.input
        encoded = (autoencoder.layers[1])(input_img)
        encoded = (autoencoder.layers[2])(encoded)
        encoded = (autoencoder.layers[3])(encoded)
        encoder = Model(input_img, encoded)
        encoder.summary()
        encoded_imgs = encoder.predict(images/255.)
        recon_imgs = autoencoder.predict(images/255.) * 255
        np.save('./{}'.format(args.read.split('/')[-1]), encoded_imgs)
        #recon_plot(images, recon_imgs, 'autoencoder')
if __name__ == "__main__":
    args = parse_args()
    main(args)
