import argparse
from keras.models import load_model
from keras.models import Model
from utils import *
import keras.backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_movie(filename):
    def genre_to_number(genres, all_genres):
        result = []
        for g in genres.split('|'):
            if g not in all_genres:
                all_genres.append(g)
            result.append( all_genres.index(g) )
        return result, all_genres

    movies, all_genres = [[]] * 3953, []
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            movieID, title, genre = line[:-1].split('::')
            genre_numbers, all_genres = genre_to_number(genre, all_genres)
            movies[int(movieID)] = genre_numbers
    
    categories = len(all_genres)
    for i, m in enumerate(movies):
        movies[i] = to_categorical(m, categories)

    print('movies:', np.array(movies).shape)
    return movies, all_genres

def to_categorical(index, categories):
    categorical = np.zeros(categories, dtype=int)
    categorical[index] = 1
    return list(categorical)





parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('--gpu_fraction', default=0.1, type=float)
parser.add_argument('--model', type=str)
args = parser.parse_args()


movies, all_genres = read_movie('data/movies.csv')

movie_list = []
for i, m in enumerate(movies):
    if np.sum(m) != 0:
        movie_list.append(i)
print('movie list:', np.array(movie_list).shape)
movie_enc = pk.load(open(movie_enc_path, 'rb'))
movie_data = np.array([np.argmax(a) for a in movie_enc.transform(np.array(movie_list).reshape(-1,1)).toarray()])
print('movie data:', np.array(movie_list).shape)

model = load_model(args.model)
get_emb_movieID = K.function([model.get_layer('input_2').input], \
                             [model.get_layer('embedding_2').output])

movie_list_reshape = movie_data.reshape(-1, 1)
movie_dim = np.squeeze(get_emb_movieID([movie_list_reshape])[0])
print("Shape of movie_dim : ", movie_dim.shape)



print('T-SNE')
tsne = TSNE(n_components=2)
movie_2d = tsne.fit_transform(X=movie_dim)


print('movie_2d:', movie_2d.shape)

print('============================================================')
print('Plot 2d Graph')

categories = []
category_index = [[0, 1, 2, 3], [5, 6, 12, 14], [4, 7, 11, 13, 17], [8, 9, 10, 16]]
colors = ['red', 'green', 'blue', 'black', 'lightgray']

for i in range(4):
    print(colors[i] + ':', np.array(all_genres)[category_index[i]] )
print('lightgray: [\'other\']')

for idx in category_index:
    category_array = np.zeros(18)
    category_array[idx] = 1
    categories.append( category_array )

def find_category(in_c):
    max_similar, color = 0, 4
    for i, c in enumerate(categories):
        similar = np.sum(c * in_c)
        if similar > max_similar:
            max_similar, color = similar, i
    return colors[color]

plt.clf()
for (ID, point) in zip(movie_list, movie_2d):
    print('\rmovie ID: %d' % ID, end='', flush=True)
    genre = movies[ID]
    plt.plot(point[0], point[1], '.', color=find_category(genre))
print('')

plt.savefig(args.model + '_emb.png', dpi=300)
plt.show()



