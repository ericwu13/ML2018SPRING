import argparse
from utils import *
from keras.layers import Reshape, Embedding, Dot, add, Input
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment classification')
    parser.add_argument('model')
    parser.add_argument('--gpu_fraction', default=0.3, type=float)
    parser.add_argument('--train_path', default='data/train.csv', type=str)
    parser.add_argument('--dim', default=18, type=int)
    return parser.parse_args()

args = parse_args()

user_train, movie_train, rating = read_data_transform(args.train_path)
# Build model

u_inp = Input(shape=(1,))
u_emb = Embedding(len(user_train), args.dim, embeddings_regularizer=l2(0.00001))(u_inp)
u = Reshape((args.dim,))(u_emb)

m_inp = Input(shape=(1,))
m_emb = Embedding(len(movie_train), args.dim, embeddings_regularizer=l2(0.00001))(m_inp)
m = Reshape((args.dim,))(m_emb)

u_bias = Embedding(len(user_train), 1, embeddings_regularizer=l2(0.00001))(u_inp)
u_bias = Reshape((1,))(u_bias)
m_bias = Embedding(len(movie_train), 1, embeddings_regularizer=l2(0.00001))(u_inp)
m_bias = Reshape((1,))(m_bias)

output = Dot(axes=-1)([u,m])
output = add([output, u_bias, m_bias])

model = Model([u_inp, m_inp], [output])
es = EarlyStopping(monitor='val_loss', patience = 15, verbose=1, mode='min')
csv = CSVLogger('log/report_NoPunc.csv')
save_dir = 'ckpt/{}/'.format(args.model)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
save_path = save_dir + "weights.{epoch:03d}-{val_loss:.5f}.h5"
ckpt = ModelCheckpoint(filepath=save_path,\
                         verbose=1, save_best_only=True,
                         monitor='val_loss', mode='min' )
model.compile(loss='mse', optimizer='adam')
model.fit([user_train, movie_train], rating, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es, csv, ckpt])

