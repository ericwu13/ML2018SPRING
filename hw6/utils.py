import numpy as np
import _pickle as pk
import pandas as pd
from para import *
import random


def read_data_transform(data_path, mode='train'):
    print('  Loadin from {}...'.format(data_path))

    user_enc = pk.load(open(user_enc_path, 'rb'))
    movie_enc = pk.load(open(movie_enc_path, 'rb'))
    
    data = pd.read_csv(data_path).values
    user_data = np.array([np.argmax(a) for a in user_enc.transform(data[:,1].reshape(-1,1)).toarray()])
    movie_data = np.array([np.argmax(a) for a in movie_enc.transform(data[:,2].reshape(-1,1)).toarray()])
    if(mode == 'train'):
        label = np.array(data[:,3])
        return user_data, movie_data, label
    else:
        return user_data, movie_data

def split_data(data, fraction=0.1):
    length = len(data[0])
    points = int(length * fraction)
    pairs = list(zip(data[0], data[1], data[2]))
    random.shuffle(pairs)
    user, movie, rating = zip(*pairs)
    user = np.array(user)
    movie = np.array(movie)
    rating = np.array(rating)
    return user[:length-points], movie[:length-points], rating[:length-points],\
            user[length-points:], movie[length-points:], rating[length-points:]
