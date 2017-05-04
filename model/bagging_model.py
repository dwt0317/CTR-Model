import numpy as np
import constants
from sklearn import metrics

test_y_file = constants.dir_path + "sample\\test.Y"
lr_out_file = constants.project_path + "model/m_lr/model_file/lr.out"
fm_out_file = constants.project_path + "model/m_fm/model_file/fm.out"

# 不好使，model相关性太强
def baggging():
    test_y = np.loadtxt(open(test_y_file), dtype=int)
    lr_out = np.loadtxt(open(lr_out_file), dtype=float)
    fm_out = np.loadtxt(open(fm_out_file), dtype=float)
    print lr_out.shape, fm_out.shape
    bag_pred = lr_out * 0.2 + fm_out * 0.8
    auc_test = metrics.roc_auc_score(test_y, bag_pred)
    print auc_test

if __name__ == '__main__':
    baggging()
