# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from scipy import io
from scipy import sparse
import datetime
import Constants

training_X_file = Constants.dir_path + "sample\\embedding\\training.X4.embedding"
training_Y_file = Constants.dir_path + "sample\\training.Y"
test_X_file = Constants.dir_path + "sample\\embedding\\test.X4.embedding"
test_Y_file = Constants.dir_path + "sample\\test.Y"


# read coo format matrix from file
def readcoo(filename):
    training = open(filename, "r")
    fields = training.readline().strip('\n').split(' ')
    rows = int(fields[0])
    columns = int(fields[1])
    values = int(fields[2])
    print rows, columns, values
    b = sparse.lil_matrix((rows, columns), dtype=float)
    i = 0
    for line in training:
        field = line.strip('\n').split(' ')
        row = field[0]
        column = field[1]
        value = field[2]
        b[int(row), int(column)] = float(value)
        if i % 1000000 == 0:
            print i
        i += 1
    return b


def lr():
    begin = datetime.datetime.now()
    classifier = LogisticRegression(n_jobs=2)
    # training_x = np.loadtxt(open(training_Y_file), dtype=int)
    training_x = readcoo(training_X_file)
    training_y = np.loadtxt(open(training_Y_file), dtype=int)
    print "Loading data completed."
    #
    classifier.fit(training_x, training_y)
    print "Training completed."

    # cross_val_score(classifier, training_x, training_y, cv=10)
    # print "Cross validation completed."

    joblib.dump(classifier, "train_model_norm_clean" + ".pkl", compress=3)      #加个3，是压缩，一般用这个

    # classifier = joblib.load("train_model_2.pkl")

    test_x = readcoo(test_X_file)
    test_y = np.loadtxt(open(test_Y_file), dtype=int)
    y_pred = classifier.predict(test_x)

    score = metrics.accuracy_score(test_y, y_pred)
    # prob_train = classifier.predict_proba(training_x)[:, 1]  # proba得到两行，一行错的一行对的
    # auc_train = metrics.roc_auc_score(training_y, prob_train)
    prob_test = classifier.predict_proba(test_x)[:, 1]  # proba得到两行, 一行错的一行对的,对的是点击的概率，错的是不点的概率
    auc_test = metrics.roc_auc_score(test_y, prob_test)

    end = datetime.datetime.now()
    log_file = open("result/lr_baseline", "a")
    log_file.write("id + ctr + similarity + norm + clean:" + '\n')
    log_file.write("score: " + str(score) + '\n')
    log_file.write("auc_test: " + str(auc_test) + '\n')
    log_file.write("time: " + str(end - begin) + '\n' + '\n')
    log_file.close()


if __name__ == '__main__':

    lr()

    # test_y = np.loadtxt(open(test_Y_file), dtype=int)
    # pred_y_file = "D:\\Dvlp_workspace\\SAD\\libfm-1.40.windows\\output2.libfm"
    # pred_y = np.loadtxt(open(pred_y_file))
    # print test_y[:10], pred_y[:10]
    # auc_test = metrics.roc_auc_score(test_y, pred_y)
    # print auc_test
