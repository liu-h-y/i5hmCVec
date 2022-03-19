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

c_list = np.logspace(-5, 15, 21, base=2)
c_list = c_list.tolist()

g_list = np.logspace(-15, -5, 11, base=2)
g_list = g_list.tolist()

sum_tasks = len(c_list)*len(g_list)
count = 0

fo = open("res/svm.txt", 'w')
fo.write("c\tgamma\tACC\tSn\tSp\tAUC\tAP\tMCC\tTP\tTN\tFP\tFN\n")

for c in c_list:
    for g in g_list:
        sum_y_test = np.array([])
        sum_y_pre = np.array([])
        sum_y_pre_prob = np.array([])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
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

        fo.write(str(c) + "\t" + str(g) + "\t" + str(final_test_accuracy) \
                 + "\t" + str(final_test_sensitivity) + "\t" + str(final_test_specificity) + "\t" + str(final_test_auc) \
                 + "\t" +str(final_test_ap)+ "\t" + str(final_test_mcc) + "\t" + str(TP) + "\t" + str(TN)+ "\t" + str(FP)+ "\t" + str(FN)+"\n")

        count += 1
        print("progress  :"+str(count)+"of"+str(sum_tasks))
fo.close()