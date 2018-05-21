from utils.util import *
from para import *

#base_dir = os.path.dirname((os.path.realpath(__file__)))
def preprocess_trainData(dm, dim=256, maxlen=40, iteration=16, mode='reproc'):
    #corpus_path = os.path.join(base_dir, 'mydict/corpus.txt')
    #cmap_path = os.path.join(base_dir, 'mydict/cmap.pkl')
    #w2v_path = os.path.join(base_dir, 'mydict/word2vec.pkl')
    dm.load_myDict(path='mydict/stop_words_list', mode='stpWrd')
    if mode is 'train':
        dm.load_myDict(corpus_path, 'corpus')
        dm.load_myDict(cmap_path, 'cmap')
    if mode is 'retrain':
        dm.load_myDict(corpus_path, 'corpus')
        dm.load_myDict(cmap_path, 'cmap')
        dm.train_w2v(w2v_path, dim=dim, maxlen=maxlen, iteration=iteration)
    if mode is 'reproc':
        dm.init_DocNCmap()
        dm.preprocess_document()
        dm.dump_myDict(corpus_path, 'corpus')
        dm.dump_myDict(cmap_path, 'cmap')
        dm.train_w2v(w2v_path, dim=dim, maxlen=maxlen, iteration=iteration)
    #print(len(dm.cmap))
    w2v = Word2Vec.load(w2v_path)
    wordvec = dm.transformByWord2Vec(w2v, maxlen)
    length_train = len(dm.get_data('train_data')[0])
    X, Y = wordvec[:length_train], (dm.get_data('train_data'))[1]
    return (X, Y), wordvec[length_train:]

def preprocess_testData(dm, dim=256, maxlen=40):
    #cmap_path = os.path.join(base_dir, 'mydict/cmap.pkl')
    #w2v_path = os.path.join(base_dir, 'mydict/word2vec.pkl')
    dm.load_myDict(path='mydict/stop_words_list', mode='stpWrd')
    dm.init_DocNCmap(mode='test')
    dm.load_myDict(cmap_path, 'cmap')
    dm.transformBycmap()
    dm.update_cmap()
    corpus_path = os.path.join(base_dir, 'mydict/test_corpus.txt')
    dm.dump_myDict(corpus_path, 'corpus')
    w2v = Word2Vec.load(w2v_path)
    X = dm.transformByWord2Vec(w2v, maxlen=maxlen)
    return X

'''
dm = DataManager()
dm.add_rawData('train_data', 'data/training_label.txt')
dm.add_rawData('semi_data', 'data/training_nolabel.txt', False)
dm.load_myDict(path='mydict/stop_words_list', mode='stpWrd')
preprocess_trainData(dm, retrain=False)
print(dm.docoment[0])
'''

