import numpy as np
import sys
import pandas as pd
import csv

trainX_data_path = sys.argv[1]
trainY_data_path = sys.argv[2]
test_data_path = sys.argv[3]
write_data_path = sys.argv[4]
means = []
std = []

def train_data_parse(data, Type):
    global means
    global std
    if(Type == "X"):
        data = (np.array(data)).astype(float)
        #ones = np.ones((len(data), 1));
        #data = np.concatenate((data,ones), axis=1)
        means.append(np.mean(data[:,0]))
        means.append(np.mean(data[:,10]))
        means.append(np.mean(data[:,78]))
        means.append(np.mean(data[:,79]))
        means.append(np.mean(data[:,80]))
        means.append(np.mean(data[:, 78] - data[:, 79]))
        std.append(np.std(data[:,0]))
        std.append(np.std(data[:,10]))
        std.append(np.std(data[:,78]))
        std.append(np.std(data[:,79]))
        std.append(np.std(data[:,80]))
        std.append(np.std(data[:, 78] - data[:, 79]))

    data = (np.array(data)).astype(float)
    return data;

def test_data_parse(data):
    data = (np.array(data)).astype(float)
    #ones = np.ones((len(data), 1));
    #data = np.concatenate((data,ones), axis=1)
    return data;

def norm(x):
    mean = float(np.mean(x))
    std = np.std(x)
    return (x - mean)/std

def rule_feature(data):
    global means
    global std
    return data
    feat_1 = (np.power((data[:, 0], data[:, 10], data[:, 78], data[:, 79], data[:, 80]) , 2.0)).T # 3
    feat_1[:, 0] = norm(feat_1[:, 0])
    feat_1[:, 1] = norm(feat_1[:, 1])
    feat_1[:, 2] = norm(feat_1[:, 2])
    feat_1[:, 3] = norm(feat_1[:, 3])
    feat_1[:, 4] = norm(feat_1[:, 4])
    feat_2 = (np.power((data[:, 0], data[:, 10], data[:, 78], data[:, 79], data[:, 80]) , 3.0)).T # 3
    feat_2[:, 0] = norm(feat_2[:, 0])
    feat_2[:, 1] = norm(feat_2[:, 1])
    feat_2[:, 2] = norm(feat_2[:, 2])
    feat_2[:, 3] = norm(feat_2[:, 3])
    feat_2[:, 4] = norm(feat_2[:, 4])
    feat_3 = (np.power((data[:, 0], data[:, 10], data[:, 78], data[:, 79], data[:, 80]) , 4.0)).T # 3
    feat_3[:, 0] = norm(feat_3[:, 0])
    feat_3[:, 1] = norm(feat_3[:, 1])
    feat_3[:, 2] = norm(feat_3[:, 2])
    feat_3[:, 3] = norm(feat_3[:, 3])
    feat_3[:, 4] = norm(feat_3[:, 4])
    feat_4 = (np.power((data[:, 0], data[:, 10], data[:, 78], data[:, 79], data[:, 80]) , 5.0)).T # 3
    feat_4[:, 0] = norm(feat_4[:, 0])
    feat_4[:, 1] = norm(feat_4[:, 1])
    feat_4[:, 2] = norm(feat_4[:, 2])
    feat_4[:, 3] = norm(feat_4[:, 3])
    feat_4[:, 4] = norm(feat_4[:, 4])
    data[:, 0] = (data[:, 0] - means[0]) / std[0]
    data[:, 10] = (data[:, 10] - means[1]) / std[1]
    data[:, 78] = (data[:, 78] - means[2]) / std[2]
    data[:, 79] = (data[:, 79] - means[3]) / std[3]
    data[:, 80] = (data[:, 80] - means[4]) / std[4]
    return np.concatenate((data, feat_1, feat_2, feat_3, feat_4), axis=1)

def get_validation(data_x, data_y, frac):
    length = len(data_x)
    point = int(length * frac)
    return (data_x[point:,:], data_y[point:], data_x[:point,:], data_y[:point])

def sigmoid(z):
    f = 1. / (1. + np.exp(-z))
    bound = 1e-8
    return np.clip(f, bound, 1-bound)
def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

def training(model, train_data):
    global val_x
    global val_y
    train_x = train_data[0]
    train_y = train_data[1]
    train_data_size = len(train_x)
    mean1_star = np.zeros((model,))
    mean2_star = np.zeros((model,))
    sigma1_star = np.zeros((model, model))
    sigma2_star = np.zeros((model, model))
    count_1 = 0
    count_2 = 0
    # calculate means
    for i in range(train_data_size):
        if train_y[i] == 1:
            mean1_star += train_x[i]
            count_1 += 1
        else:
            mean2_star += train_x[i]
            count_2 += 1
    mean1_star /= count_1
    mean2_star /= count_2

    #calculate sigma
    for i in range(train_data_size):
        if train_y[i] == 1:
            sigma1_star += np.dot(np.transpose([train_x[i] - mean1_star]), [train_x[i] - mean1_star])
        else:
            sigma2_star += np.dot(np.transpose([train_x[i] - mean2_star]), [train_x[i] - mean2_star])

    sigma1_star /= count_1
    sigma2_star /= count_2
    shared_sigma = (float(count_1) / train_data_size) * sigma1_star + (float(count_2) / train_data_size) * sigma2_star
    return (mean1_star, mean2_star, shared_sigma, count_1, count_2)

def predict(test_x, mean1, mean2, shared_sigma, N1, N2):
    sigma_inverse = inv(shared_sigma)
    w = np.dot((mean1 - mean2), sigma_inverse)
    x = test_x.T
    b =(-0.5) * np.dot(np.dot(mean1, sigma_inverse), mean1) \
        + (0.5) * np.dot(np.dot(mean2, sigma_inverse), mean2) \
        + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y 



def get_accuracy(y_predict, test_y):
    count = 0;
    for i in range(len(y_predict)):
        if(y_predict[i] > 0.5 and test_y[i] == 1):
            count += 1
        elif(y_predict[i] < 0.5 and test_y[i] == 0):
            count += 1
    return float(count) / len(test_y)

def write_prediction(y_predict):
    f = open(write_data_path,"w")
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for i in range(len(y_predict)):
        string = str(i+1)
        if(y_predict[i] > 0.5):
            y_predict[i] = 1
        else:
            y_predict[i] = 0
        w.writerow([string, int(y_predict[i])]);
    f.close()

train_raw_x = ((pd.read_csv(trainX_data_path, header=0)).values).astype(float)
train_raw_y = ((pd.read_csv(trainY_data_path, header=-1)).values).astype(float)
train_data_x = train_data_parse(train_raw_x, 'X')
train_data_y = train_data_parse(train_raw_y, 'Y')
train_data_x = rule_feature(train_data_x)

test_raw_x = ((pd.read_csv(test_data_path)).values).astype(float)
test_x = test_data_parse(test_raw_x)
test_x = rule_feature(test_x)

(train_x, train_y, val_x, val_y) = get_validation(train_data_x, train_data_y, 0.2);

model = 123

(m1, m2, shared_sigma, n1, n2) = training(model, (train_x, train_y))
y_predict = predict(val_x, m1, m2, shared_sigma, n1, n2)

accuracy = get_accuracy(y_predict, val_y);
#print("Accuracy: "+str(accuracy))

(m1, m2, shared_sigma, n1, n2) = training(model, (train_data_x, train_data_y))
y_predict = predict(test_x, m1, m2, shared_sigma, n1, n2)

write_prediction(y_predict)

