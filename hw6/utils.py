import numpy as np
import _pickle as pk
import pandas as pd
from para import *


def read_data_transform(data_path, mode='train'):
    print('  Loadin from {}...'.format(data_path))

    user_enc = pk.load(open(user_enc_path, 'rb'))
    movie_enc = pk.load(open(movie_enc_path, 'rb'))
    
    data = pd.read_csv(data_path).values
    user_data = np.array([np.argmax(a) for a in user_enc.transform(data[:,1].reshape(-1,1)).toarray()])
    movie_data = np.array([np.argmax(a) for a in movie_enc.transform(data[:,2].reshape(-1,1)).toarray()])
    if(mode == 'train'):
        label = np.array([ x[3] for x in data])
        return user_data, movie_data, label
    else:
        return user_data, movie_data

