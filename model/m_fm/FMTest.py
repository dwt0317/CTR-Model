import pywFM
import numpy as np
import constants
import datetime
from sklearn import metrics
import pandas as pd
import cPickle as pickle
import os

project_path = "D:\\Workspaces\\SAD\\kdd\\"

train_X_file = constants.dir_path + "sample\\features\\train.new_basic.libfm"
train_Y_file = constants.dir_path + "sample\\training.Y"
test_X_file = constants.dir_path + "sample\\features\\test.new_basic.libfm"
test_Y_file = constants.dir_path + "sample\\test.Y"

cygwin_libfm = "D:/Development/cygwin64/home/dwt/libfm/bin/"


def test():
    features = np.matrix([
        #     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
        #    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
        [1, 0, 0, 1, 0, 0, 0, 0.3, 0.3, 0.3, 0, 13, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0.3, 0.3, 0.3, 0, 14, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0.3, 0.3, 0.3, 0, 16, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 5, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0.5, 0.5, 8, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0.5, 0, 0.5, 0, 9, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 12, 1, 0, 0, 0]
    ])
    target = [5, 3, 1, 4, 5, 1, 5]
    fm = pywFM.FM(task='regression', num_iter=5)

    # split features and target for train/test
    # first 5 are train, last 2 are test
    model = fm.run(features[:5], target[:5], features[5:], target[5:])

    print(model.predictions)
    # you can also get the model weights
    print(model.weights)
    prob_test = model.predictions
    auc_test = metrics.roc_auc_score(target[5:], prob_test)
    print auc_test

    # print model.pairwise_interactions.shape
    # with open(constants.dir_path + "sample\\features\\train_test.fm.np", 'w') as f:
    #     for line in model.pairwise_interactions:
    #         np.savetxt(f, line, fmt='%.4f')


# build fm interaction vectors
def build_fm_interaction():
    begin = datetime.datetime.now()
    test_y = np.loadtxt(open(test_Y_file), dtype=int)
    fm = pywFM.FM(task='classification', num_iter=100, learning_method='mcmc', temp_path=project_path+"model\\m_fm\\tmp\\")

    model = fm.run(None, None, None, None, train_path=train_X_file, test_path=test_X_file,
                   model_path=project_path + "model\\m_fm\\model_file\\fm_model",
                   out_path=project_path + "model\\m_fm\\model_file\\fm.out"
                   )
    end = datetime.datetime.now()

    print model.pairwise_interactions.shape
    y_pred = model.predictions
    auc_test = metrics.roc_auc_score(test_y, y_pred)
    accuracy = metrics.accuracy_score(test_y, y_pred)
    logloss = metrics.log_loss(test_y, y_pred)
    np.savetxt(open(constants.project_path + "result/10_9_fm_pred", "w"), y_pred, fmt='%.5f')

    rcd = str(end) + '\n'
    rcd += "fm: new basic" + '\n'
    rcd += "accuracy: " + str(accuracy) + '\n'
    rcd += "logloss: " + str(logloss) + '\n'
    rcd += "auc_test: " + str(auc_test) + '\n'
    rcd += "time: " + str(end - begin) + '\n' + '\n'
    print rcd

    log_file = open(project_path + "result/oct_result", "a")
    log_file.write(rcd)
    log_file.close()

    print model.pairwise_interactions.shape
    # with open(constants.dir_path + "sample\\features\\fm_features\\interactions.fm_sparse_id.np", 'w') as f:
    #     for line in model.pairwise_interactions:
    #         np.savetxt(f, line, fmt='%.4f')
    # with open(constants.dir_path + "sample\\features\\fm_features\\prediction.fm.np", 'w') as f:
    #     for line in model.predictions:
    #         np.savetxt(f, line, fmt='%.4f')


def load_cygwin_pred():
    y_prob = np.loadtxt(open(cygwin_libfm+"/out/fm_no-cl-im-comb_45.out"), dtype=float)

    test_y = np.loadtxt(open(test_Y_file), dtype=float)
    auc_test = metrics.roc_auc_score(test_y, y_prob)
    # accuracy = metrics.accuracy_score(test_y, y_pred)
    logloss = metrics.log_loss(test_y, y_prob)

    rcd = str(datetime.datetime.now()) + '\n'
    rcd += "fm: basic_no-cl-im-comb_45" + '\n'
    # rcd += "accuracy: " + str(accuracy) + '\n'
    rcd += "logloss: " + str(logloss) + '\n'
    rcd += "auc_test: " + str(auc_test) + '\n' + '\n'
    print rcd

    log_file = open(project_path + "result/oct_result", "a")
    log_file.write(rcd)
    log_file.close()


if __name__ == '__main__':
    # build_fm_interaction()
    # test()
    load_cygwin_pred()