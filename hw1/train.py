import numpy as np;
import matplotlib.pyplot as plt
import sys
import csv
import math
np.set_printoptions(precision=4,suppress=True)
###################
### Global Data ###
###################
train_data_path = "ml-2018spring-hw1/train.csv"
test_data_path = sys.argv[1];
#write_data_path = sys.argv[2];
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

#wi = np.ones((actual_feature*train_hour+1, 1));
#wi = np.array(wi, dtype=np.float)

point = int(iteration/1000)
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

def rule_data(data, index):
    data = np.delete(data, 10, 0)
    for i in range(train_hour+1):
        #tmp = data[:, index+i].astype(float);
        tmp = np.array(data[:, index+i], dtype=np.float64);
        #if(not tmp.any()):
        if((0 in tmp) or (tmp[9] > 500)):
            return index+i+1;
    return index;

def rule_feature1(data):
    feat_1 = np.multiply(data[16,:].astype(float), (np.sin((data)[15,:].astype(float) * np.pi / 180. )))
    feat_2 = np.multiply(data[3:7,:].astype(float), data[3:7,:].astype(float));
    data = (np.delete(data, [0,1,2,11,14,15,16,17], 0))
    return np.concatenate((data.flatten(), feat_1.flatten(), feat_2.flatten()), axis=0);
def rule_feature2(data):
    feat_1 = (np.sin((data)[15,:].astype(float) * np.pi / 180. ))
    feat_2 = np.multiply(data[3:7,:].astype(float), data[3:7,:].astype(float));
    data = (np.delete(data, [0,1,2,11,14,15,16,17], 0))
    return np.concatenate((data.flatten(), feat_1.flatten(), feat_2.flatten()), axis=0);

def train_data_form1(data):
    data_align = data_alignment(data);
    tmp_x = []
    tmp_y = []
    for i in range(0, 24*day_n*12, 480):
        for j in range(day_n*24):
            index = (rule_data(data_align, i+j) - i)
            if(j != index):
                j = index - 1
                continue
            if(j+1+train_hour < day_n*24):
                data = rule_feature1(data_align[:, i+j:i+j+train_hour]);
                tmp_x.append(np.append(data, [1]));
                #tmp_x.append(np.append(data_align[:, i+j:i+j+train_hour].flatten(), [1]));
                tmp_y.append(data_align[9][i+j+train_hour]);
            else:
                break;
    tmp_x = np.array(tmp_x)
    tmp_y = np.array(tmp_y)
    #tmp_x = tmp_x.astype(float)
    tmp_x = np.array(tmp_x, dtype=np.float);
    #tmp_y = tmp_y.astype(float)
    tmp_y = np.array(tmp_y, dtype=np.float);
    #print(len(tmp_y))
    return (tmp_x, tmp_y);

def train_data_form2(data):
    data_align = data_alignment(data);
    tmp_x = []
    tmp_y = []
    for i in range(0, 24*day_n*12, 480):
        for j in range(day_n*24):
            index = (rule_data(data_align, i+j) - i)
            if(j != index):
                j = index - 1
                continue
            if(j+1+train_hour < day_n*24):
                data = rule_feature2(data_align[:, i+j:i+j+train_hour]);
                tmp_x.append(np.append(data, [1]));
                #tmp_x.append(np.append(data_align[:, i+j:i+j+train_hour].flatten(), [1]));
                tmp_y.append(data_align[9][i+j+train_hour]);
            else:
                break;
    tmp_x = np.array(tmp_x)
    tmp_y = np.array(tmp_y)
    #tmp_x = tmp_x.astype(float)
    tmp_x = np.array(tmp_x, dtype=np.float);
    #tmp_y = tmp_y.astype(float)
    tmp_y = np.array(tmp_y, dtype=np.float);
    #print(len(tmp_y))
    return (tmp_x, tmp_y);
def test_data_form1(data):
    data_align = data_alignment(data);
    tmp = [];
    for i in range(0, len(data_align[0]), train_hour):
        data = rule_feature1(data_align[:, i:i+train_hour])
        tmp.append(np.append(data, [1]));
        #tmp.append(np.append(data_align[:,i:i+train_hour].flatten(), [1]));
    tmp = np.array(tmp)
    tmp = tmp.astype(float)
    #print(len(tmp_x[1]))
    #print((tmp[1]))
    return tmp

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

def derivative(train_x, train_y, wi):
    # get model output
    y_star = np.dot(train_x, wi);
    # training y
    train_y = (np.reshape(train_y, np.shape(y_star)))
    y_head = train_y;
    error = y_head - y_star;
    tmp_w = ((-2)*np.dot(train_x.T, error)) + 2*regular_factor*wi;
    #print(len(tmp_w));
    return tmp_w;

def LossFunction(x, y, wi):
    y_star = np.dot(x, wi);
    errors = y - y_star;
    return (np.sqrt((np.mean(errors**2))), regular_factor*np.sum(wi**2) / (len(wi)-1)//train_hour)

def validation(x, y, w):
    return LossFunction(x, y, w);

def training(model, train_data):
    train_x = train_data[0]
    train_y = train_data[1]
    wi = np.ones((model*train_hour+1, 1));
    wi = np.array(wi, dtype=np.float)
    lr = 1
    lr_w = np.zeros((model*train_hour+1, 1))
    check = True;
    i = 0;
    loss = []
    while check:
        y_star = np.dot(train_x, wi);
        y_head = (np.reshape(train_y, np.shape(y_star)))
        error = y_head - y_star;

        w_grad = ((-2)*np.dot(train_x.T, error)) + 2*regular_factor*wi;
        lr_w = lr_w + lr*np.multiply(w_grad, w_grad)
        w_new = wi - np.multiply(np.reciprocal(np.sqrt(lr_w)), w_grad)

        check = ((np.mean(abs(wi - w_new))) >= 10**(-6))
        (l,r) = (np.sqrt((np.mean(error**2))), regular_factor*np.sum(wi**2) / model)
        loss.append(l);
        print ("train times:"+str(i)+" Error: "+str(l+r)+" ",end="")
        print ("({0}, {1})".format(l, r)+"\r",end="")
        i += 1
        wi = w_new
    return (wi, loss)

####################
### data reading ###
####################
train_data = read_file(train_data_path);
train_data = data_parse("train", train_data);
train_data = np.array(train_data);

test_data = read_file(test_data_path);
test_data = data_parse("test", test_data);
test_data = np.array(test_data);
#print(test_data);

####################
### data forming ###
####################
#(train_x, train_y) = train_data_form(train_data);
#(data_x_1, data_y_1) = (train_x[frac:,:], train_y[frac:])
#(data_x_2, data_y_2) = (train_x[:frac, :], train_y[:frac])

(tmp1, tmp_y) = train_data_form1(train_data)
(tmp2, tmp_y) = train_data_form2(train_data)
frac = len(tmp1)//100
train_y = tmp_y[frac:]
val_y = tmp_y[:frac]

train_x_for1 = tmp1[frac:,:]
val_x_for1 = tmp1[:frac,:]
train_x_for2 = tmp2[frac:,:]
val_x_for2 = tmp2[:frac,:]
#print(len(train_x)) #print(count)
test_x_1 = test_data_form1(test_data);
test_x_2 = test_data_form2(test_data);

################
### training ###
################




model1 = (feature_1)
model2 = (feature_2)

'''
print("Model1 training")
(w1, loss1) = training(model1, (train_x_for1, train_y));
model_1_loss = validation(val_x_for1, val_y, w1)
print('\n'+"Validation loss: " + str(model_1_loss[0]));

print("Model2 training")
(w2, loss2) = training(model2, (train_x_for2, train_y));
model_2_loss = validation(val_x_for2, val_y, w2)
print('\n'+"Validation loss: " + str(model_2_loss[0]));

print("Average error for model 1: " + str((model_1_loss[0])))
#print("Average error for model 2: " + str((model_2_loss[0])))
'''

filename = 'model';
(w, loss) = training(model1, (tmp1, tmp_y));
np.save(filename, w)


'''
#Writing CSV
f = open(write_data_path,"w")
w = csv.writer(f)
w.writerow(["id", "value"])
for i in range(len(y_predict)):
    string = "id_" + str(i)
    w.writerow([string, float(y_predict[i])]);
f.close()
'''

#plotting
loss = np.array(loss)
plt.plot(np.arange(0, len(loss)-10), (loss[10:]))
#plt.show()
