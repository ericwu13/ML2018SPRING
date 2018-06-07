from preprocess import *
from keras.models import load_model
import argparse
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pandas as pd


parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('--test_path', default='data/testing_data.txt', type=str)
parser.add_argument('--result_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--max_length', default=40, type=int)
args = parser.parse_args()

def get_session(gpu_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#K.set_session(get_session(args.gpu_fraction))

dm = DataManager()
dm.add_rawData('test_data', args.test_path, False)
test_X = preprocess_testData(dm, maxlen=args.max_length)
model = load_model(args.model_path)
model.summary()
predict_Y = model.predict(test_X, batch_size=1024, verbose=True)

print(predict_Y)
ID = []
label = []
for i in range(len(predict_Y)):
    ID.append(i)
    if(predict_Y[i] > 0.5): 
        label.append(1)
    else:
        label.append(0)
columns = ['id', 'label']
d = np.array([ID,label])
df = pd.DataFrame(data=d.T, columns=columns)
df.to_csv(args.result_path, encoding='utf-8', index=False)
