import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing,metrics
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description="Train and evaluation i5hmCVec on svm")

parser.add_argument('--s', type=str)
parser.add_argument('--K',type=int,nargs='+')
parser.add_argument('--c',type=float)
parser.add_argument('--g',type=float)


args = parser.parse_args()

species = args.s
k_list = args.K
c = args.c
g = args.g

if species == 'dm':
    df = pd.read_csv("./feature/dm_feature.csv",header=None)
if species == 'mouse':
    df = pd.read_csv("./feature/mouse_feature.csv",header=None)

select_index = []
for k in k_list:
    select_index.extend(range((k-3)*100+1,(k-2)*100+1))

X = df.iloc[:, select_index]
y = df.iloc[:, 0]
X = X.values
y = y.values

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

        # print c,g
        clf = SVC(C=c, gamma=g, probability=True)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_predict_prob = clf.predict_proba(X_test)
        y_predict_prob = y_predict_prob[:, 1]
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