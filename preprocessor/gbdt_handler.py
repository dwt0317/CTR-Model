import constants
import cPickle as pickle
train_path = constants.dir_path + "sample\\features\\gbdt_features\\train.onehot.dict"
validation_path = constants.dir_path + "sample\\features\\gbdt_features\\validation.onehot.dict"
test_path = constants.dir_path + "sample\\features\\gbdt_features\\test.onehot.dict"


def build_gbdt(dataset):
    path = ""
    if dataset == "train":
        path = train_path
    elif dataset == "test":
        path = test_path
    else:
        path = validation_path
    gbdt_feature = pickle.load(open(path, "rb"))
    print gbdt_feature[1]
    return gbdt_feature


if __name__ == '__main__':
    build_gbdt("test")