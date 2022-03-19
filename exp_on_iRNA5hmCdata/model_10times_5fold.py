import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("Feature/feature.csv", header=None)
X = df.iloc[:,1:]
y = df.iloc[:,0]
X = X.values
y = y.values

c = 1.0
g = 0.03125

fo = open("res/svm_5fold_10times.txt", 'w')
fo.write("c\tgamma\tACC\tSn\tSp\tAUC\tAP\tMCC\tTP\tTN\tFP\tFN\n")

for random_num in range(10):
    sum_y_test = np.array([])
    sum_y_pre = np.array([])
    sum_y_pre_prob = np.array([])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_num)
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

    fo.write(str(final_test_accuracy) + "\t" + str(final_test_sensitivity) + "\t" + str(final_test_specificity) + "\t" + str(final_test_auc) \
             + "\t" +str(final_test_ap)+ "\t" + str(final_test_mcc) + "\t" + str(TP) + "\t" + str(TN)+ "\t" + str(FP)+ "\t" + str(FN)+"\n")


fo.close()