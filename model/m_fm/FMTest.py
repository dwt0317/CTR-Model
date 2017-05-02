import pywFM
import numpy as np
from Utils import read_coordinate_mtx
import constants
import datetime
from sklearn import metrics


project_path = "D:\\Workspaces\\SAD\\kdd\\"

train_x_file = project_path + "model\\m_fm\\dataset\\train.gbdt_no_id.libfm"
train_y_file = constants.dir_path + "sample\\training.Y"
validation_x_file = project_path + "model\\m_fm\\dataset\\validation.gbdt_no_id.libfm"
test_x_file = project_path + "model\\m_fm\\dataset\\test.gbdt_no_id.libfm"
test_y_file = constants.dir_path + "sample\\test.Y"




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
    target = [0, 1, 1, 0, 1, 0, 1]

    fm = pywFM.FM(task='c', num_iter=20, learning_method='sgd', temp_path=project_path + "tmp\\")
    print features[:5]
    # split features and target for train/test
    # first 5 are train, last 2 are test
    model = fm.run(features[:5], target[:5], features[5:], target[5:], train_path=None, test_path=None)
    prob_test = model.predictions

    auc_test = metrics.roc_auc_score(target[5:], prob_test)
    print auc_test
    print model.pairwise_interactions.shape
    with open(constants.dir_path + "sample\\features\\train_test.fm.np", 'w') as f:
        for line in model.pairwise_interactions:
            np.savetxt(f, line, fmt='%.4f')


# build fm interaction vectors
def build_fm_interaction():
    begin = datetime.datetime.now()
    test_y = np.loadtxt(open(test_y_file), dtype=int)
    fm = pywFM.FM(task='classification', num_iter=100, learning_method='mcmc', temp_path=project_path+"model\\m_fm\\tmp\\")

    model = fm.run(None, None, None, None, train_path=train_x_file, test_path=test_x_file)
    end = datetime.datetime.now()

    print model.pairwise_interactions.shape
    prob_test = model.predictions
    auc_test = metrics.roc_auc_score(test_y, prob_test)
    print auc_test

    log_file = open(project_path + "result/lr_baseline", "a")
    log_file.write("fm + 620 dimensions + 100 iters:" + '\n')
    log_file.write("auc_test: " + str(auc_test) + '\n')
    log_file.write("time: " + str(end - begin) + '\n' + '\n')
    log_file.close()

    print model.pairwise_interactions.shape
    with open(constants.dir_path + "sample\\features\\interactions.fm.np", 'w') as f:
        for line in model.pairwise_interactions:
            np.savetxt(f, line, fmt='%.4f')

if __name__ == '__main__':
    build_fm_interaction()
