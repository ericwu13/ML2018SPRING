from keras.models import load_model
from utils import io
import sys
import numpy as np
import csv

te_feats = io.read_dataset('test', False)
y_prob = 0;
num_model = int(sys.argv[1])
name = sys.argv[2]
for i in range(num_model):
    model = load_model(sys.argv[i+3])
    prob = model.predict(te_feats)
    y_prob += prob

y_prob = y_prob / float(num_model)
y_classes = y_prob.argmax(axis=-1)

with open("predict/ensemble{}.csv".format(str(num_model) + "_" + name), 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(te_feats)):
        csv_writer.writerow([i]+[y_classes[i]])
