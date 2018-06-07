import os
base_dir = os.path.dirname((os.path.realpath(__file__)))
corpus_path = os.path.join(base_dir, 'model/corpus.txt')
cmap_path = os.path.join(base_dir, 'model/cmap.pkl')
w2v_path = os.path.join(base_dir, 'model/word2vec.pkl')
token_path = os.path.join(base_dir, 'model/token.pkl')
