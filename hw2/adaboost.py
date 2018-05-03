import numpy as np
import sys
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# age: 1
# workclass: 8+1
# fnlwgt: 1
# education: 16
# education-num: 16
# marital-status: 7
# occupation: 14+1
# relationship: 6
# race: 5
# sex: 2
# capital-gain: 1
# capital-loss: 1
# hours-per-week: 1
# native-country: 41+1
# 1+9+1+16+16+7+15+6+5+2+1+1+1+42+16
###################
### Global Data ###
###################
trainX_data_path = sys.argv[1]
trainY_data_path = sys.argv[2]
test_data_path = sys.argv[3]
depth = int(sys.argv[4])
est = int(sys.argv[5])
iteration = 0
regular_factor = 0

train_raw_x, train_raw_y = [], []
train_x, train_y = [], []
train_having_uknown = []
#7, 54, 116
test_raw_x = []
test_x = []

(means, std) = ([], [])

def read_file(path):
    data = []
    f = open(path, newline = '', encoding = 'latin-1');
    for row in csv.reader(f):
        tmp = [];
        for i in range(len(row)):
                if(type(row[i]) is str):
                    tmp.append((row[i]));
                else:
                    tmp.append(float(row[i]));
        data.append(tmp);
    f. close();
    return data

def train_data_parse(data, Type):
    global means
    global std
    if(Type == "X"):
        data.pop(0);
        data = (np.array(data)).astype(float)
        data = np.delete(data, train_having_uknown, axis=0)
        ones = np.ones((len(data), 1));
        data = np.concatenate((data,ones), axis=1)
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
    else:
        data = np.delete(data, train_having_uknown, axis=0)
    data = (np.array(data)).astype(float)
    return data;

def test_data_parse(data):
    data.pop(0);
    data = (np.array(data)).astype(float)
    ones = np.ones((len(data), 1));
    data = np.concatenate((data,ones), axis=1)
    return data;

def norm(x):
    mean = float(np.mean(x))
    std = np.std(x)
    return (x - mean)/std

def rule_feature(data):
    global means
    global std
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

def sigmoid(z):
    f = 1. / (1. + np.exp(-z))
    bound = 1e-8
    return np.clip(f, bound, 1-bound)

def training(model, train_data):
    global val_x
    global val_y
    train_x = train_data[0]
    train_y = train_data[1]
    #wi = np.zeros((model + 1, 1));
    wi = np.reshape(np.array([1e-4 for x in range(model+1)]), (model+1,1))
    #wi = np.array(wi, dtype=np.float)
    best_w = np.zeros((model + 1, 1));
    best_acc = 0;
    #lr = 5*1e-3 # Good lr
    #lr = 8*1e-3 # Good lr
    #lr = 5e-12
    lr = 1e-2
    lr_w = np.zeros((model+1, 1))
    i = 0;
    train_loss = []
    val_loss = []
    while i < iteration:
        z = (np.dot(train_x, wi))
        y_star = sigmoid(z);
        y_head = train_y

        w_grad = ((-1)*np.dot(train_x.T, y_head - y_star)) + 2*regular_factor*wi
        #lr_w = lr_w + np.multiply(w_grad, w_grad)
        #w_new = wi - lr*np.multiply(np.reciprocal(np.sqrt(lr_w)), w_grad)
        w_new = wi - (lr/(i+1))*w_grad

        l = -np.mean(y_head * np.log(y_star + 10**(-10)) + (1 - y_head) * np.log(1 - y_star + 10**(-10)))
        r = regular_factor*np.sum(w_new**2);

        z = (np.dot(val_x, wi))
        val_y_star = sigmoid(z);
        val_l = -np.mean(val_y * np.log(val_y_star + 10**(-10)) + (1 - val_y) * np.log(1 - val_y_star + 10**(-10)))

        train_loss.append(l);
        val_loss.append(val_l)
        accuracy = get_accuracy(val_x, val_y, w_new);
        print ("train times:"+str(i)+" accuracy: "+str(accuracy)+"",end="")
        #print ("train times:"+str(i)+" ",end="")
        print ("({0}, {1})".format(l, r)+"\r",end="")
        if(best_acc < accuracy):
            #print("Best Accuracy: " + str(accuracy) , end="")
            best_w = w_new.copy()
            best_acc = accuracy

        i += 1
        wi = w_new
    return (best_w, train_loss, val_loss)

def get_validation(data_x, data_y, frac):
    length = len(data_x)
    point = int(length * frac)
    return (data_x[point:,:], data_y[point:], data_x[:point,:], data_y[:point])

def get_accuracy(test_x, test_y, w):
    y_predict = np.dot(test_x, w)
    count = 0;
    for i in range(len(y_predict)):
        if(y_predict[i] >= 0.5 and test_y[i] == 1):
            count += 1
        elif(y_predict[i] < 0.5 and test_y[i] == 0):
            count += 1
    return float(count) / len(test_y)

def lossfunction(x, y, w):
    class_1 = y;
    class_2 = 1 - y;
    y = (np.concatenate((class_1, class_2), axis=1))
    z = (np.dot(x, w))
    f_1 = sigmoid(z);
    f_2 = 1 - f_1
    f = np.concatenate((f_1, f_2), axis=1);

    tmp = (np.multiply(np.log(f + 10**(-10)), y))

    l = np.mean((tmp[0,:]+tmp[1,:]))
    return -l;

def write_prediction(y_predict, depth, est):
    f = open("depth_{}_n_estimator_{}.csv".format(depth, est),"w")
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for i in range(len(y_predict)):
        string = str(i+1)
        if(y_predict[i][0] >= 0.5):
            w.writerow([string, 0]);
        else:
            w.writerow([string, 1]);
    f.close()

train_raw_x = read_file(trainX_data_path)
train_raw_y = read_file(trainY_data_path)
train_data_x = train_data_parse(train_raw_x, 'X')
train_data_y = train_data_parse(train_raw_y, 'Y')
train_data_x = rule_feature(train_data_x)

test_raw_x = read_file(test_data_path)
test_x = test_data_parse(test_raw_x)
test_x = rule_feature(test_x)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         algorithm="SAMME",
                         n_estimators=est)

bdt.fit(train_data_x, train_data_y.ravel())

y = bdt.predict_proba(test_x)
print(y)
write_prediction(y, depth, est)

#(train_x, train_y, val_x, val_y) = get_validation(train_data_x, train_data_y, 0.2);
#model = 143

#(w, loss, val_loss) = training(model, (train_data_x, train_data_y))
#length = np.arange(len(loss))



