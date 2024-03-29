from keras.models import load_model, Input, Model
from keras.layers import average
from utils import io
import sys
import numpy as np
import csv

def ensembleModels(models, name):
    y_model = []
    for i in models:
        y_model.append(load_model(i))

    model_input = Input(shape=y_model[0].input_shape[1:])
    print(y_model[0].input_shape)

    yModels=[model(model_input) for model in y_model]
    yAvg=average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    modelEns.save(name+'.h5')
    return modelEns

te_feats = io.read_dataset('test', False)
num_model = int(sys.argv[1])
name = sys.argv[2]
models = []
for i in range(num_model):
    models.append(sys.argv[i+3])

model = ensembleModels(models, name)
y_prob = model.predict(te_feats)
y_classes = y_prob.argmax(axis=-1)

with open("predict/ensemble{}.csv".format(str(num_model) + "_" + name), 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(te_feats)):
        csv_writer.writerow([i]+[y_classes[i]])
