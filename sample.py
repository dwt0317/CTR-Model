import constants

# 采样
def sample():
    file_read = open(constants.dir_path + "training.txt")
    file_train = open(constants.dir_path + "sample\\training.part", 'w')
    file_valid = open(constants.dir_path + "sample\\validation.part", 'w')
    file_test = open(constants.dir_path + "sample\\test.part", 'w')
    i = 0
    for line in file_read:
        # file_train.write(line)
        if i % 100000 == 0 :
            print i
        if i < 1800000:
            file_train.write(line)
        elif i >= 1800000 and i < 2000000:
            file_valid.write(line)
        else:
            file_test.write(line)
        if i > 2200000 :    # 多了一行数据
            file_read.close()
            file_train.close()
            file_valid.close()
            file_test.close()
            break
        i = i + 1