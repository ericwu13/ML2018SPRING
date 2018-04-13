from utils import io
from keras.models import load_model
import csv
import numpy as np
import sys

def testing(X, model, path):
    global model_name
    prd_class = model.predict_classes(X)
    print(prd_class)
    path = path.split('/')[1]
    with open("predict/{}.csv".format(path), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(X)):
            csv_writer.writerow([i]+[prd_class[i]])

model_name = sys.argv[1]
model = load_model(model_name)
te_feats = io.read_dataset('test', False)
testing(te_feats, model, model_name)


