import argparse
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from utils import *
from keras.layers import Reshape, Embedding, dot, add, Input
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment classification')
    parser.add_argument('model')
    parser.add_argument('--gpu_fraction', default=0.1, type=float)
    parser.add_argument('--train_path', default='data/train.csv', type=str)
    parser.add_argument('--dim', default=18, type=int)
    return parser.parse_args()

def get_session(gpu_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
args = parse_args()
K.set_session(get_session(args.gpu_fraction))

user_train, movie_train, rating = read_data_transform(args.train_path)
# Build model

print(len(np.unique(user_train)))
print(len(np.unique(movie_train)))
print(rating.shape)
input()
user_train, movie_train, rating_train, user_val, movie_val, rating_val = split_data((user_train, movie_train, rating), 0.2)
u_inp = Input(shape=(1,))
u_emb = Embedding(6040, args.dim, embeddings_regularizer=l2(0.00001))(u_inp)
u = Reshape((args.dim,))(u_emb)

m_inp = Input(shape=(1,))
m_emb = Embedding(3688, args.dim, embeddings_regularizer=l2(0.00001))(m_inp)
m = Reshape((args.dim,))(m_emb)

u_bias = Embedding(6040, 1, embeddings_regularizer=l2(0.00001))(u_inp)
u_bias = Reshape((1,))(u_bias)
m_bias = Embedding(3688, 1, embeddings_regularizer=l2(0.00001))(m_inp)
m_bias = Reshape((1,))(m_bias)

output = dot([u, m], axes=-1)
output = add([output, u_bias, m_bias])

model = Model([u_inp, m_inp], [output])
model.summary()
es = EarlyStopping(monitor='val_loss', patience = 15, verbose=1, mode='min')
csv = CSVLogger('log/mf.csv')

save_dir = 'ckpt/{}/'.format(args.model)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
save_path = save_dir + "weights.{epoch:03d}-{val_loss:.5f}.h5"

ckpt = ModelCheckpoint(filepath=save_path,\
                         verbose=1, save_best_only=True,
                         monitor='val_loss', mode='min' )
model.compile(loss='mse', optimizer='adam')
model.fit([user_train, movie_train], [rating_train], epochs=100, batch_size=512, validation_data=([user_val, movie_val], [rating_val]), callbacks=[es, csv, ckpt])

