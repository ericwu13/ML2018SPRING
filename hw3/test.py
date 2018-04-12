from utils import io
from keras.models import load_model
import csv
import numpy as np

model_name = "weights.100-0.6829.h5"
batch_size = 128

def testing(X, model):
    global model_name
    #y_test = model.predict(X, batch_size=128, verbose=1)
    #prd_class = np.argmax(y_test, axis=1)
    prd_class = model.predict_classes(X)
    print(prd_class)
    with open("predict/{}2.csv".format(model_name), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(X)):
            csv_writer.writerow([i]+[prd_class[i]])


te_feats = io.read_dataset('test', False)
tr_feats, tr_labels = io.read_dataset()
model = load_model("ckpt/"+model_name)
#score = model.evaluate(tr_feats, tr_labels)
#print("\n"+str(score[1]))
testing(te_feats, model)
