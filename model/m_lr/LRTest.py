# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV
import datetime
import constants
from Utils import read_coo_mtx

training_X_file = constants.dir_path + "sample\\features\\training.no_id.coor"
training_Y_file = constants.dir_path + "sample\\training.Y"
test_X_file = constants.dir_path + "sample\\features\\test.no_id.coor"
test_Y_file = constants.dir_path + "sample\\test.Y"


def lr():
    begin = datetime.datetime.now()
    grid = False
    train_x = read_coo_mtx(training_X_file)
    train_y = np.loadtxt(open(training_Y_file), dtype=int)
    test_x = read_coo_mtx(test_X_file)
    test_y = np.loadtxt(open(test_Y_file), dtype=int)

    print "Loading data completed."
    print "Read time: " + str(datetime.datetime.now() - begin)
    classifier = LogisticRegression(solver='sag', random_state=8)
    if grid:
        param_grid = {'C': [1, 5, 10]}
        grid = GridSearchCV(estimator=classifier, scoring='roc_auc', param_grid=param_grid)
        grid.fit(train_x, train_y)
        print "Training completed."
        print grid.cv_results_
        print grid.best_estimator_

    if not grid:
        classifier.fit(train_x, train_y)
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
        rcd = "no_id + fm + gbdt:" + '\n'
        rcd += "score: " + str(score) + '\n'
        rcd += "auc_test: " + str(auc_test) + '\n'
        rcd += "time: " + str(end - begin) + '\n' + '\n'
        print rcd
        log_file = open(constants.project_path+"result/lr_baseline", "a")
        log_file.write(rcd)
        log_file.close()


if __name__ == '__main__':
    lr()

    # test_y = np.loadtxt(open(test_Y_file), dtype=int)
    # pred_y_file = "D:\\Dvlp_workspace\\SAD\\libfm-1.40.windows\\output2.libfm"
    # pred_y = np.loadtxt(open(pred_y_file))
    # print test_y[:10], pred_y[:10]
    # auc_test = metrics.roc_auc_score(test_y, pred_y)
    # print auc_test
