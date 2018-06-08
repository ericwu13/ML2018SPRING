from para import *
from utils import *
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import argparse
import keras.backend.tensorflow_backend as K
import tensorflow as tf


parser = argparse.ArgumentParser(description='MF')
parser.add_argument('--test_path', default='data/test.csv', type=str)
parser.add_argument('--train_path', default='data/train.csv', type=str)
parser.add_argument('--result_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
args = parser.parse_args()


def get_session(gpu_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



K.set_session(get_session(args.gpu_fraction))

test_users, test_movies = read_data_transform(args.test_path, mode='test')

model = load_model(args.model_path)
model.summary()

predict = model.predict([test_users, test_movies], batch_size=1024, verbose=True)

print(predict)
fuck = []
for i in range(len(test_users)):
    tmp = []
    tmp.append(i+1)
    tmp.append(np.squeeze(predict[i]))
    fuck.append(tmp)

columns = ['TestDataID', 'Rating']
df = pd.DataFrame(data=fuck, columns=columns)
df[['TestDataID']].astype('str')
df.to_csv(args.result_path, encoding='utf-8', index=False)
