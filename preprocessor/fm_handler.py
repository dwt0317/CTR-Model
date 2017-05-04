import constants
import numpy as np

fm_features_path = constants.dir_path + "sample\\features\\fm_features\\interactions.fm.np"


def build_fm_features():
    v = np.loadtxt(fm_features_path)[:20]
    return v

if __name__ == '__main__':
    c = build_fm_features()
    print c
