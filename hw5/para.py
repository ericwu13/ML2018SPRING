import os
base_dir = os.path.dirname((os.path.realpath(__file__)))
corpus_path = os.path.join(base_dir, 'model/corpus_3W_30padd.txt')
cmap_path = os.path.join(base_dir, 'model/cmap_3W_30padd.pkl')
w2v_path = os.path.join(base_dir, 'model/word2vec_3W_30padd.pkl')
token_path = os.path.join(base_dir, 'model/token.pkl')
