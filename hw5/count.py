import _pickle as pk
from gensim.models import Word2Vec

f = open('mydict/cmap.pkl', 'rb')
w2v = Word2Vec.load('mydict/word2vec_2W.pkl')


cmap = pk.load(f)

dic = {}

for key, value in cmap.items():
    if value not in dic:
        dic[value] = value


#print(dic)
print(len(cmap))
print(len(dic))

#print(w2v)
