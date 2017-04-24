# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import datetime
import constants

training_X_file = constants.dir_path + "sample\\features\\training.sparse_id.coor"
training_Y_file = constants.dir_path + "sample\\training.Y"
test_X_file = constants.dir_path + "sample\\features\\test.sparse_id.coor"
test_Y_file = constants.dir_path + "sample\\test.Y"


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
    grid = False
    training_x = readcoo(training_X_file)
    training_y = np.loadtxt(open(training_Y_file), dtype=int)
    test_x = readcoo(test_X_file)
    test_y = np.loadtxt(open(test_Y_file), dtype=int)

    print "Loading data completed."
    classifier = LogisticRegression(n_jobs=2)
    if grid:
        param_grid = {'C': [1, 5, 10]}
        grid = GridSearchCV(estimator=classifier, scoring='roc_auc', param_grid=param_grid)
        grid.fit(training_x, training_y)
        print "Training completed."
        print grid.cv_results_
        print grid.best_estimator_

    if not grid:
        classifier.fit(training_x, training_y)
        # cross_val_score(classifier, training_x, training_y, cv=10)
        # print "Cross validation completed."
        # joblib.dump(classifier, "train_model_norm_clean" + ".pkl", compress=3)      #加个3，是压缩，一般用这个
        # classifier = joblib.load("train_model_2.pkl")

        y_pred = classifier.predict(test_x)

        score = metrics.accuracy_score(test_y, y_pred)
        # prob_train = classifier.predict_proba(training_x)[:, 1]  # proba得到两行，一行错的一行对的

        prob_test = classifier.predict_proba(test_x)[:, 1]  # proba得到两行, 一行错的一行对的,对的是点击的概率，错的是不点的概率
        auc_test = metrics.roc_auc_score(test_y, prob_test)

        end = datetime.datetime.now()
        log_file = open("result/lr_baseline", "a")
        log_file.write("sparse_id + cos + discrete_sim:" + '\n')
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
