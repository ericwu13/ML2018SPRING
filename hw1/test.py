import numpy as np
import sys
import csv

test_data_path = sys.argv[1]
write_data_path = sys.argv[2]
w = np.load('model.npy')
regular_factor = 10000
feature_n = 18
feature_1 = 15
feature_2 = 15
iteration = 100000
train_hour = 9
day_n = 20

count = 0;
train_data = [];
test_data = [];

train_x = [];
train_y = [];
test_x = [];
test_ans = [];
def read_file(path):
    data = []
    f = open(path, newline = '', encoding = 'latin-1');
    for row in csv.reader(f):
        tmp = [];
        for i in range(len(row)):
            if(row[i] == "NR"):
                tmp.append(0)
            else:
                if(type(row[i]) is str):
                    tmp.append((row[i]));
                else:
                    tmp.append(float(row[i]));
        data.append(tmp);
    f. close();
    return data

def data_parse(Type, data):
    if(Type == "train"):
        data.pop(0);
        for i in range(len(data)):
            data[i] = data[i][3:];
        return data;
    elif(Type == "test"):
        for i in range(len(data)):
            data[i] = data[i][2:];
        return data

def data_alignment(data):
    data_align = data[:feature_n]
    for i in range(feature_n, len(data), feature_n):
      data_align = np.concatenate((data_align, data[i:i+feature_n]), axis=1);
    return data_align;

def test_data_form2(data):
    data_align = data_alignment(data);
    tmp = [];
    for i in range(0, len(data_align[0]), train_hour):
        data = rule_feature2(data_align[:, i:i+train_hour])
        tmp.append(np.append(data, [1]));
        #tmp.append(np.append(data_align[:,i:i+train_hour].flatten(), [1]));
    tmp = np.array(tmp)
    tmp = tmp.astype(float)
    #print(len(tmp_x[1]))
    #print((tmp[1]))
    return tmp

def rule_feature2(data):
    feat_1 = np.multiply(data[16,:].astype(float), (np.sin((data)[15,:].astype(float) * np.pi / 180. )))
    feat_2 = np.multiply(data[3:7,:].astype(float), data[3:7,:].astype(float));
    data = (np.delete(data, [0,1,2,11,14,15,16,17], 0))
    return np.concatenate((data.flatten(), feat_1.flatten(), feat_2.flatten()), axis=0);
test_data = read_file(test_data_path);
test_data = data_parse("test", test_data);
test_data = np.array(test_data);

test_x_2 = test_data_form2(test_data);

y_predict = np.dot(test_x_2, w)
y_predict = y_predict.astype(float);

f = open(write_data_path,"w")
w = csv.writer(f)
w.writerow(["id", "value"])
for i in range(len(y_predict)):
    string = "id_" + str(i)
    w.writerow([string, float(y_predict[i])]);
f.close()
