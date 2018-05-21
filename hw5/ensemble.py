from utils.util import *
from keras.models import load_model, Input, Model
from keras.layers import average
import sys
import numpy as np

def ensembleModels(models):
    y_model = []
    count = 0
    for i in models:
        model = load_model(i)
        model.name = 'model_{}'.format(count)
        y_model.append(model)
        count += 1

    model_input = Input(shape=y_model[0].input_shape[1:])
    print(y_model[0].input_shape)

    yModels=[model(model_input) for model in y_model]
    yAvg=average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    modelEns.save('ensemble.h5')
    return modelEns

models = []
num_model = int(sys.argv[1])
for i in range(num_model):
    models.append(sys.argv[i+2])

ensembleModels(models)
