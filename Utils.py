# -*- coding:utf-8 -*-
from scipy import sparse
import random


def list2dict(mylist):
    myDict = {}
    i = 0
    for v in mylist:
        myDict[v] = i
        i += 1
    return myDict


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# read coo format matrix from file
def read_coordinate_mtx(filename):
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


# shuffle data before sgd
def shuffle(training_x, training_y):
    m = training_x.shape[0]
    for i in range(m):
        swapIdx = random.randint(0, m)
        tmpX = training_x[i]
        tmpY = training_y[i]
        training_x[i] = training_x[swapIdx]
        training_y[i] = training_y[swapIdx]
        training_x[swapIdx] = tmpX
        training_y[swapIdx] = tmpY
    print "Shuffle finished."