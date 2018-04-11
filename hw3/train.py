import os
import numpy as np
import random 
import sys
from keras.models import load_model
from keras.utils import plot_model

height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 150
zoom_range = 0.2
isValid = 1
model_name = "ckpt/weights.038-0.64716.h5"

def read_dataset(mode='train', isFeat=True):
  data = []
  cnt = 0
  with open(os.path.join('data/', '{}.csv'.format(mode))) as f:
    for line_id, line in enumerate(f):
      if cnt == 0:
        cnt += 1
        continue
      feat = []
      if isFeat:
        label, feat = line.split(',')
      else:
      	_, feat = line.split(',')
      feat = np.fromstring(feat, dtype=int, sep=' ')/255
      feat = np.reshape(feat, (48, 48, 1))

      if isFeat:
      	data.append((feat, int(label), line_id))
      else:
      	data.append(feat)

  if isFeat:
    feats, labels, line_ids = zip(*data)
  else:
    feats = data
  feats = np.asarray(feats)

  if isFeat:
    labels = to_categorical(np.asarray(labels, dtype=np.int32))
    return feats, labels, line_ids
  else:
    return feats
def testing(X, model):
  global model_name
  #ans = model.predict(X, batch_size=batch_size,verbose=1)
  ans = model.predict_classes(X)
  ans = list(ans)
  print('\n')
  with open('{}.csv'.format(model_name), 'w', encoding='big5') as f:
    f.write('id,label\n')
    for i in range(len(ans)):
      p = np.argmax(ans[i])
      f.write(repr(i) + ',' + repr(p) + '\n')


def main():
  global model_name
  te_feats = read_dataset('test', False)
  model = load_model(model_name)
  testing(te_feats, model)
main() 
