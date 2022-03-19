import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy

def get_model(input_size):
    model = Sequential()
    model.add(Conv1D(filters=32,kernel_size=3,padding='valid',activation='relu',input_shape=(input_size,1)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(20,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))

    return model

parser = argparse.ArgumentParser(description="Train and evaluation i5hmCVec on cnn")

parser.add_argument('--s', type=str)
parser.add_argument('--K', type=int, nargs='+')
parser.add_argument('--lr', type=float)
parser.add_argument('--epoch', type=int)

args = parser.parse_args()

species = args.s
k_list = args.K
learning_rate = args.lr
epoch = args.epoch

if species == 'dm':
    df = pd.read_csv("./feature/dm_feature.csv",header=None)
if species == 'mouse':
    df = pd.read_csv("./feature/mouse_feature.csv",header=None)

select_index = []
for k in k_list:
    select_index.extend(range((k - 3) * 100 + 1, (k - 2) * 100 + 1))

X = df.iloc[:, select_index]
y = df.iloc[:, 0]
X = X.values
y = y.values

feature_num = X.shape[1]

acc = []
sn = []
sp = []
auc = []
ap = []
mcc = []
for random_state_num in range(10):
    sum_y_test = np.array([])
    sum_y_pre = np.array([])
    sum_y_pre_prob = np.array([])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_num)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        scaler.fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], -1, 1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], -1, 1).astype('float32')

        clf = get_model(feature_num)

        clf.compile(loss=BinaryCrossentropy(), optimizer=SGD(lr=learning_rate),
                    metrics=['binary_accuracy'])

        clf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch,
                batch_size=16, verbose=1)

        y_predict_prob = clf.predict(X_test)
        y_predict = np.around(y_predict_prob)
        sum_y_test = np.append(sum_y_test, y_test)
        sum_y_pre = np.append(sum_y_pre, y_predict)
        sum_y_pre_prob = np.append(sum_y_pre_prob, y_predict_prob)

    TN, FP, FN, TP = (metrics.confusion_matrix(sum_y_test, sum_y_pre)).ravel()
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)
    TP = float(TP)

    final_test_accuracy = (TP + TN) / (TN + FP + FN + TP)
    final_test_sensitivity = TP / (TP + FN)
    final_test_specificity = TN / (FP + TN)
    final_test_auc = metrics.roc_auc_score(sum_y_test, sum_y_pre_prob)
    final_test_ap = metrics.average_precision_score(sum_y_test, sum_y_pre_prob)
    final_test_mcc = metrics.matthews_corrcoef(sum_y_test, sum_y_pre)

    acc.append(final_test_accuracy)
    sn.append(final_test_sensitivity)
    sp.append(final_test_specificity)
    auc.append(final_test_auc)
    ap.append(final_test_ap)
    mcc.append(final_test_mcc)

acc = np.array(acc)
sn = np.array(sn)
sp = np.array(sp)
auc = np.array(auc)
ap = np.array(ap)
mcc = np.array(mcc)

print('acc')
print(np.mean(acc))
print(np.std(acc))

print('sn')
print(np.mean(sn))
print(np.std(sn))

print('sp')
print(np.mean(sp))
print(np.std(sp))

print('auc')
print(np.mean(auc))
print(np.std(auc))

print('ap')
print(np.mean(ap))
print(np.std(ap))

print('mcc')
print(np.mean(mcc))
print(np.std(mcc))