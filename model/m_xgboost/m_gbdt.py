# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics   #Additional scklearn functions
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier
import cPickle as pickle
import constants
import xgboost as xgb
import os
import datetime


# tp, fn, fp, tn
def get_metric(test_y, train_pred):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(test_y)):
        pred_y = 0
        if train_pred[i] > 0.5:
            pred_y = 1
        if test_y[i] == 1 and pred_y == 1:
            tp += 1
        elif test_y[i] == 1 and pred_y == 0:
            fn += 1
        elif test_y[i] == 0 and pred_y == 1:
            fp += 1
        elif test_y[i] == 0 and pred_y == 0:
            tn += 1
    print tp, fn, fp, tn


def train_model():
    title = 'nn_no-cl-im-user-comb_t50'
    begin = datetime.datetime.now()
    train_x = constants.dir_path + "sample\\features\\train."+title+".libfm"
    train_y = np.loadtxt(constants.dir_path + "sample\\training.Y", dtype=int)
    test_x = constants.dir_path + "sample\\features\\test."+title+".libfm"
    test_y = np.loadtxt(constants.dir_path + "sample\\test.Y", dtype=int)
    validation_x = constants.dir_path + "sample\\features\\validation.gbdt.libfm"

    # svmlight格式自带label
    # train_data = load_svmlight_file(train_x)

    rounds = 100
    classifier = XGBClassifier(learning_rate=0.1, n_estimators=rounds, max_depth=3,
                               min_child_weight=1, gamma=0, subsample=0.8,
                               objective='binary:logistic', nthread=2)

    grid = False
    # if grid:
    #     param_test1 = {
    #         'max_depth': range(3, 5, 2),
    #         'min_child_weight': range(1, 6, 3)
    #     }
    #     gsearch = GridSearchCV(estimator=classifier, param_grid=param_test1, scoring='roc_auc', n_jobs=2)
    #     gsearch.fit(train_data[0].toarray(), train_data[1])
    #     print gsearch.best_params_, gsearch.best_score_

    if not grid:
        train_set = xgb.DMatrix(train_x)
        print "train done"
        validation_set = xgb.DMatrix(test_x)
        print "test done"
        watchlist = [(train_set, 'train'), (validation_set, 'eval')]
        params = {"objective": 'binary:logistic',
                  "booster": "gbtree",
                  'eval_metric': 'logloss',
                  "eta": 0.1,
                  "max_depth": 8,
                  'silent': 0,
                  'subsample': 0.9,
                  'min_child_weight': 2,
                  'nthread': 2,
                  }
        print "Training model..."
        xgb_model = xgb.train(params, train_set, rounds, watchlist, verbose_eval=True)
        y_pred = xgb_model.predict(xgb.DMatrix(test_x))
        auc_test = metrics.roc_auc_score(test_y, y_pred)
        logloss = metrics.log_loss(test_y, y_pred)

        end = datetime.datetime.now()
        day = datetime.date.today()
        np.savetxt(open(constants.project_path + "result/pred/"+title+"_gbdt_pred"+str(day), "w"), y_pred, fmt='%.5f')

        rcd = str(end) + '\n'
        rcd += "gbdt: "+ title + '\n'
        rcd += str(params) + '\n'
        rcd += "logloss: " + str(logloss) + '\n'
        rcd += "auc_test: " + str(auc_test) + '\n'
        rcd += "time: " + str(end - begin) + '\n' + '\n'
        print rcd
        log_file = open(constants.project_path+"result/oct_result", "a")
        log_file.write(rcd)
        log_file.close()


        # pickle.dump(xgb_model, open(os.getcwd()+"/gbdt_model", "wb"))
        # print "dump model finished"
        # test_ind = xgb_model.predict(xgb.DMatrix(test_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
        # train_ind = xgb_model.predict(xgb.DMatrix(train_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)



def onehot_feature():
    print "load_data"
    onehot = []
    print "transform"
    gbdt_feature = pickle.load(open(constants.dir_path + "sample\\features\\gbdt_features\\validation.idx", "rb"))

    for line in gbdt_feature:
        temp_onehot = []
        i = 0
        for item in line:
            temp_onehot.append(int(item) + i*20 - 1)
            i += 1
        onehot.append(temp_onehot)

    pickle.dump(onehot, open(constants.dir_path + "sample\\features\\gbdt_features\\validation.onehot.dict", "wb"))


def test_read():
    gbdt_feature = pickle.load(open(constants.dir_path + "sample\\features\\gbdt_features\\train.onehot.dict", "rb"))
    for i in range(3):
        print gbdt_feature[i]

if __name__ == '__main__':
    train_model()
    # onehot_feature()

