from utils import io
from keras.models import load_model
import csv
import numpy as np
import sys

def testing(X, model, path):
    global model_name
    y_prob = model.predict(X)
    prd_class = y_prob.argmax(axis=-1)
    print(prd_class)
    #path = path.split('/')[1]
    with open(path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(X)):
            csv_writer.writerow([i]+[prd_class[i]])

model_name = sys.argv[2]
model = load_model(model_name)
#attr = np.load('attr.npy')
te_feats = io.read_dataset('test', False, path=True, data_path=sys.argv[1])
testing(te_feats, model, sys.argv[3])


