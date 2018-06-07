import os
import re
import numpy as np
import _pickle as pk
from gensim.models import Word2Vec, KeyedVectors #from gensim.corpora import Dictionary
from pprint import pprint
from keras.preprocessing.text import Tokenizer
from para import *

def get_freqDict(document):
    freq = {}
    for s in document:
        for w in s:
            if w in freq: freq[w] += 1
            else:         freq[w] = 1
    return freq

def convert_punc(cmap, document, orin_document):
    print('  Converting punctuations...')
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            tmp = w
            if w == '': continue
            excCnt, queCnt, dotCnt = w.count('!'), w.count('?'), w.count('.')
            if queCnt >=4:    tmp = '_???'
            elif queCnt >= 2: tmp = '_??'
            elif queCnt >= 1: tmp = '_?'
            elif excCnt >= 4: tmp = '_!!!'
            elif excCnt >= 2: tmp = '_!!'
            elif excCnt >= 1: tmp = '_!'
            elif dotCnt >= 2: tmp = '_...'
            elif dotCnt >= 1: tmp = '_.'
            cmap[orin_document[i][j]] = tmp
            s[j] = tmp
        document[i] = s

def convert_headDuplicates(cmap, document, orin_document, minfreq=800):
    print('  Converting head duplication...')
    count = 0
    freq = get_freqDict(document)
    f = open('mydict/log/head.txt', 'w', encoding='utf_8')
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            if freq[w] > minfreq: continue
            tmp = re.sub(r'^(([a-z])\2+)', r'\g<2>', w)
            if(tmp != w):
                count += 1
                f.write('convert {} to {}\n'.format(w, tmp))
                s[j] = tmp
                cmap[orin_document[i][j]] = tmp
        document[i] = s
    print('    Transfer counts = {}'.format(count))

def convert_inlineDuplicates(cmap, document, orin_document, minfreq=800):
    # inline duplication occurrs frequently
    print('  Converting inline duplication...')
    count = 0
    freq = get_freqDict(document)
    f = open('mydict/log/inline.txt', 'w', encoding='utf_8')
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            if w[0] is '_': continue
            if freq[w] > minfreq: continue
            # assert head and tail don't have duplication
            #        only one duplication exists in line
            tmp = re.sub(r'(([a-z])\2{1,})', r'\g<2>', w) 
            value = freq.get(tmp, 0)
            if value < 100: continue

            if(tmp != w):
                count += 1
                f.write('convert {} to {}\n'.format(w, tmp))
                cmap[orin_document[i][j]] = tmp
                s[j] = tmp
        document[i] = s
    print('    Transfer counts = {}'.format(count))

def convert_tailDuplicates(cmap, document, orin_document, minfreq=800):
    # tails duplication occurrs the most frequently
    print('  Converting tail duplication...')
    count = 0
    freq = get_freqDict(document)
    f = open('mydict/log/tail.txt', 'w', encoding='utf_8')
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            if w[0] is '_': continue
            if freq[w] > minfreq: continue
            tmp = re.sub(r'(([a-z])\2+)$', r'\g<2>', w)
            value = freq.get(tmp, 0)
            if value < 100: continue
            if(tmp != w):
                count += 1
                f.write('convert {} to {}\n'.format(w, tmp))
                cmap[orin_document[i][j]] = tmp
                s[j] = tmp
        document[i] = s
    print('    Transfer counts = {}'.format(count))

def convert_singleAlpha(cmap, document, orin_document, minfreq=-1):
    print('  Converting single alphabet to \'_sg\'...')
    freq = get_freqDict(document)
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            if w is 't' or 'n': continue
            value = freq.get(w, 0)
            if value > minfreq: continue
            if(len(w) ==1):
                s[j] = '_sg'
                cmap[w] = s[j]
        document[i] = s

def convert_slang(cmap, document, orin_document):
    print('  Converting slang...')
    freq = get_freqDict(document)
    count = 0
    f = open('mydict/log/slang.txt', 'w', encoding='utf_8')
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            tmp = ''
            if w == '': continue
            if w[0] == '_': continue
            if w == 'u': tmp = 'you'
            elif w == 'yu': tmp = 'you'
            elif w == 'yur': tmp  = 'your'
            elif w == 'ur': tmp = 'your'
            elif w == 'dis': tmp = 'this'
            elif w == 'dat': tmp = 'that'
            elif w == 'luv': tmp = 'love'
            elif w == '2': tmp = 'to'
            else :
                w1 = re.sub(r'in$', r'ing', w)
                w2 = re.sub(r'n$', r'ing', w)
                f0, f1, f2 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0)
                fm = max(f0, f1, f2)
                if fm == f0:   continue
                elif fm == f1: tmp = w1;
                else:          tmp = w2;
            if( tmp != w): 
                #print("{} convert to {}".format(orin_document[i][j], tmp))
                f.write('convert {} to {}\n'.format(w, tmp))
                count += 1
                cmap[orin_document[i][j]] = tmp
                s[j] = tmp
        document[i] = s
    print('    Transfer counts = {}'.format(count))
def convert_rareWords(cmap, document, orin_document, minfreq=8):
    print('  Converting rare words to \'_r\'')
    count = 0
    freq = get_freqDict(document)
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            if w[0] is '_': continue
            if freq[w] > minfreq: continue
            cmap[orin_document[i][j]] = '_r'
            s[j] = '_r'
            count += 1
        document[i] = s
    print('    Transfer counts = {}'.format(count))

def convert_singularForm(cmap, document, orin_document, minfreq=1000):
    print('  Converting plural form to singular form...')
    freq = get_freqDict(document)
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is '': continue
            value = freq.get(w, 0)
            if value > minfreq: continue
            w1 = re.sub(r's$', r'', w)
            w2 = re.sub(r'es$', r'', w)
            w3 = re.sub(r'ies$', r'', w)
            f0, f1, f2, f3 = freq.get(w,0), freq.get(w1,0), \
                             freq.get(w2,0), freq.get(w3,0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            elif fm == f2: s[j] = w2;
            else:          s[j] = w3;
            cmap[w] = s[j]
        document[i] = s

def padding(document, maxlen=40):
    print('  Padding to length {}...'.format(maxlen))
    padd_doc = document
    for i, s in enumerate(document):
        padd_doc[i] = s[:40]
    maxlinelen = 0
    for i, s in enumerate(padd_doc):
        maxlinelen = max(len(s), maxlinelen)
    maxlinelen = max(maxlinelen, maxlen)
    for i, s in enumerate(padd_doc):
        padd_doc[i] = (['_pad'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return padd_doc

def cmapRefine(cmap, document, orin_document):
    print('  Refineing conversion map...')
    count = 0
    cmap['itt'] = 'it'
    cmap['luvin'] = 'loving'
    cmap['lovee'] = cmap['loove'] = cmap['looove'] = cmap['loooove'] = cmap['looooove'] \
        = cmap['loooooove'] = cmap['loves'] = cmap['loved'] = cmap['wuv'] = cmap['loooovee']\
        = cmap['loovee'] = cmap['luve'] = cmap['lov'] = cmap['luvs'] = cmap['luv']\
        = cmap['luvd'] = 'love'
    cmap['2day'] = 'today'
    cmap['2moz'] = cmap['2morow'] = cmap['2morrow'] = 'tomorrow'
    cmap['dam'] = 'damn'
    for i, s in enumerate(document):
        for j, w in enumerate(s):
            if w is 'itt': s[j] = 'it'
            if w is 'luvin': s[j] = 'loving'
            if w is 'lovee': s[j] = 'love'
            if w is 'loove': s[j] = 'love'
            if w is 'looove': s[j] = 'love'
            if w is 'loooove': s[j] = 'love'
            if w is 'looooove': s[j] = 'love'
            if w is 'loooooove': s[j] = 'love'
            if w is 'loves': s[j] = 'love'
            if w is 'loved': s[j] = 'love'
            if w is 'wuv': s[j] = 'love'
            if w is 'loovee': s[j] = 'love'
            if w is 'luve': s[j] = 'love'
            if w is 'lov': s[j] = 'love'
            if w is 'luvs': s[j] = 'love'
            if w is 'luv': s[j] = 'love'
            if w is 'luvd': s[j] = 'love'
            if w is '2day': s[j] = 'today'
            if w is '2moz': s[j] = 'tomorrow'
            if w is '2morow': s[j] = 'tomorrow'
            if w is '2morrow': s[j] = 'tomorrow'
            if w is 'dam': s[j] = 'damn'
            if w.isdigit(): 
                cmap[orin_document[i][j]] = '_n'
                s[j] = '_n'
            if w is not s[j]: count += 1
        document[i] = s
    print('    Transfer counts = {}'.format(count))



class DataManager:
    def __init__(self):
        self.data = {}              #raw data
        self.stpWrds_list = []
        self.cmap = {}              # conversion map
        self.document = []          # slightly preprocessed & tokenized data
        self.orin_document = []
    # Read data from data_path without any data preprocessing
    #   name       : string, name of data
    #   with_label : bool, read data with label or without label
    def add_rawData(self, name, data_path, with_label=True):
        print ('  Loading data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    if(name == 'test_data'):
                        lines = line.strip('\n1234567890, ')
                        if(lines == 'id,text'): pass
                        else:                   X.append(lines)
                    else:   
                        X.append(line)
        if with_label:  self.data[name] = [X,Y]
        else:           self.data[name] = [X]
    # Get data by name
    def get_data(self,name):
        return self.data[name]

    def get_semi_data(self, semi_data, label, threshold) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        print(label)
        semi_data = np.squeeze(semi_data)
        index = (label>1-threshold) + (label<threshold)
        semi_X = semi_data
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        return semi_X[index][:], semi_Y[index]

    # split data to two part by a specified ratio
    #   name  : string, same as add_data
    #   ratio : float, ratio to split
    def split_data(self, data, ratio):
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
    # Build dictionary
    #   load/save mydictionary
    #     three modes: corpus, cmap, stpWrd
    def load_myDict(self, path, mode='corpus'):
        if mode is 'corpus':
            print('  Loading', path + '...')
            with open(path, 'r', encoding='utf_8') as f:
                for line in f:
                    self.document.append(line.split())
        if mode is 'cmap':
            print('  Loading', path + '...')
            with open(path, 'rb') as f:
                self.cmap = pk.load(f)
        if mode is 'stpWrd':
            print('  Loading', path + '...')
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
                for s in lines:
                    self.stpWrds_list.append(s.lower())

    def dump_myDict(self, path, mode):
        if mode is 'corpus':
            print('  Dumping', path + '...')
            with open(path, 'w', encoding='utf_8') as f:
                for line in self.document:
                    f.write(' '.join(line) + '\n')
        if mode is 'cmap':
            print('  Dumping', path + '...')
            #print(len(self.cmap))
            with open(path, 'wb') as f:
                pk.dump(self.cmap, f)
    #   proproess document & update document and cmap
    def preprocess_document(self):
        convert_punc(self.cmap, self.document, self.orin_document)
        convert_headDuplicates(self.cmap, self.document, self.orin_document)
        convert_tailDuplicates(self.cmap, self.document, self.orin_document)
        convert_inlineDuplicates(self.cmap, self.document, self.orin_document)
        #convert_singleAlpha(self.cmap, self.document, self.orin_document)
        #convert_singularForm(self.cmap, self.document, self.orin_document)
        convert_slang(self.cmap, self.document, self.orin_document)
        convert_rareWords(self.cmap, self.document, self.orin_document)
        cmapRefine(self.cmap, self.document, self.orin_document)
        self.document = [[word for word in s if word not in (self.stpWrds_list)]\
            for s in self.document]
        self.update_cmap()

    #   train word2vec model:
    def train_w2v(self, path, dim=256, maxlen=40, iteration=16):
        padd_doc = padding(self.document, maxlen)
        #pprint(padd_doc[0])
        print('  Training word2vec with corpus size ({}, {})...'.format(len(self.document), maxlen))
        model = Word2Vec(padd_doc, size=dim, min_count=5, iter=iteration, workers=16)
        print(model)
        model.save(path)

    def transformByWord2Vec(self, w2v, maxlen=40):
        # return a word vector set
        vector = padding(self.document, maxlen)
        self.dump_myDict('mydict/padd_testCorpus.txt', 'corpus')
        '''
        for idx, s in enumerate(self.document):
            self.document[idx] = [word for word in s if word != '']
        '''

        print('  Transforming by word2vec...')
        for i, s in enumerate(self.document):
            for j, w in enumerate(s):
                if w in w2v.wv:
                    vector[i][j] = w2v.wv[w]
                else:
                    vector[i][j] = w2v.wv['_r']
        return vector 
    def transformBycmap(self):
        # transform document by cmap
        for i, s in enumerate(self.document):
            for j, w in enumerate(s):
                if w in self.cmap: s[j] = self.cmap[w]
                else: s[j] = '_r'
            self.document[i] = s
        self.document = [[word for word in s if word not in (self.stpWrds_list)]\
            for s in self.document]
        #print(self.document[944])

    def transformByBOW(self, load=False):
        print('  Creating tokenizer...')
        for i, s in enumerate(self.document):
            self.document[i] = ' '.join(s)
        #print(self.document[0])
        if load:
            print ('Load tokenizer from %s'%token_path)
            tokenizer = pk.load(open(token_path, 'rb'))
        else:
            tokenizer = Tokenizer(num_words=30000)#, reserve_zero=False)
            tokenizer.fit_on_texts(self.document)
            print ('save tokenizer to %s'%token_path)
            pk.dump(tokenizer, open(token_path, 'wb'))
        bag_of_word =  tokenizer.texts_to_matrix(self.document[:int(len(self.data['train_data'][0]))],mode='count')
        #bag_of_word =  bag_of_word + tokenizer.texts_to_matrix(self.document[int(len(self.document)/2):],mode='count')
        #print(bag_of_word.shape)
        return bag_of_word


    #   init cmap & document with data(train+semi)
    def init_DocNCmap(self, mode='train'):
        print("  Initializing conversion map...")
        document = []
        if mode is 'train':
            for key in self.data:
                if key is 'train_data' or 'semi_data':
                    document += self.data[key][0]
        else:
            for key in self.data:
                if key is 'test_data':
                    document += self.data[key][0]

        # remove stop words & useless punctuation
        punc = '. ! ? , $ ` @ # % ^ & _ - + = | \" | \\ / \' :'
        texts = [[word for word in s.split() if word[0] not in (self.stpWrds_list)+punc.split()]\
                for s in document]
        self.document = [x[:] for x in texts]
        #print(self.document[0])
        self.orin_document = [x[:] for x in texts]
        for s in self.document:
            for w in s:
                self.cmap[w] = w
        print('    Conversion map size: ', len(self.cmap))
    
    def update_cmap(self):
        cmap = {}
        for s in self.document:
            for w in s:
                cmap[w] = w
        print('    New Conversion map size: ', len(cmap))
        return len(cmap)
        
    #   caculating words frequency given document
