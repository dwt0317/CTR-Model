# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import GridSearchCV
import datetime
import constants
from Utils import read_coo_mtx
import joblib

title = 'nn_no-cl-im-comb_t15'
training_X_file = constants.dir_path + "sample\\features\\train."+title+".libfm"
training_Y_file = constants.dir_path + "sample\\training.Y"
test_X_file = constants.dir_path + "sample\\features\\test."+title+".libfm"
test_Y_file = constants.dir_path + "sample\\test.Y"


def lr():
    begin = datetime.datetime.now()
    grid = False
    # train_x = read_coo_mtx(training_X_file)
    # train_y = np.loadtxt(open(training_Y_file), dtype=int)
    # test_x = read_coo_mtx(test_X_file)
    # test_y = np.loadtxt(open(test_Y_file), dtype=int)

    train_x, train_y = load_svmlight_file(training_X_file, n_features=178348)
    test_x, test_y = load_svmlight_file(test_X_file, n_features=178348)
    # print train_x.shape, test_x.shape
    print "Loading data completed."
    print "Read time: " + str(datetime.datetime.now() - begin)

    classifier = LogisticRegression(penalty='l2', max_iter=130, tol=1e-5)
    # if grid:
    #     param_grid = {'C': [1, 5, 10]}
    #     grid = GridSearchCV(estimator=classifier, scoring='roc_auc', param_grid=param_grid)
    #     grid.fit(train_x, train_y)
    #     print "Training completed."
    #     print grid.cv_results_
    #     print grid.best_estimator_

    if not grid:
        classifier.fit(train_x, train_y)
        # cross_val_score(classifier, training_x, training_y, cv=10)
        # print "Cross validation completed."
        # joblib.dump(classifier, "new_basic_lr" + ".pkl", compress=3)      #加个3，是压缩，一般用这个
        # classifier = joblib.load("new_basic_lr.pkl")

        y_pred = classifier.predict(test_x)

        accuracy = metrics.accuracy_score(test_y, y_pred)
        # prob_train = classifier.predict_proba(training_x)[:, 1]  # proba得到两行，一行错的一行对的
        prob_test = classifier.predict_proba(test_x)[:, 1]  # proba得到两行, 一行错的一行对的,对的是点击的概率，错的是不点的概率
        logloss = metrics.log_loss(test_y, prob_test)
        auc_test = metrics.roc_auc_score(test_y, prob_test)

        end = datetime.datetime.now()
        day = datetime.date.today()
        np.savetxt(open(constants.project_path+"result/pred/"+title+"_lr_pred_"+str(day), "w"), prob_test, fmt='%.5f')

        rcd = str(end) + '\n'
        rcd += "lr: "+title+" 130" + '\n'
        rcd += str(classifier.get_params()) + '\n'
        rcd += "accuracy: " + str(accuracy) + '\n'
        rcd += "logloss: " + str(logloss) + '\n'
        rcd += "auc_test: " + str(auc_test) + '\n'
        rcd += "time: " + str(end - begin) + '\n' + '\n' + '\n'
        print rcd
        log_file = open(constants.project_path+"result/oct_result", "a")
        log_file.write(rcd)
        log_file.close()


if __name__ == '__main__':
    lr()

    # test_x, test_y = datasets.load_svmlight_file(test_X_file)
    # np.savetxt(open(constants.project_path + "result/10.7_lr_result", "w"), test_y, fmt='%.5f')
    # print test_x.shape, test_y.shape
    # test_y = np.loadtxt(open(test_Y_file), dtype=int)
    # pred_y_file = "D:\\Dvlp_workspace\\SAD\\libfm-1.40.windows\\output2.libfm"
    # pred_y = np.loadtxt(open(pred_y_file))
    # print test_y[:10], pred_y[:10]
    # auc_test = metrics.roc_auc_score(test_y, pred_y)
    # print auc_test
