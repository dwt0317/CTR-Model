# -*- coding:utf-8 -*-
from scipy import sparse
import random
import numpy as np


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
def read_coo_mtx(filename):
    training = open(filename, "r")
    fields = training.readline().strip('\n').split(' ')
    row_num = int(fields[0])
    column_num = int(fields[1])
    value_num = int(fields[2]) + 10
    print row_num, column_num, value_num
    rows = np.empty(value_num, dtype=int)
    columns = np.empty(value_num, dtype=int)
    values = np.empty(value_num, dtype=float)
    i = 0
    for line in training:
        field = line.strip('\n').split(' ')
        rows[i] = int(field[0])
        columns[i] = int(field[1])
        values[i] = float(field[2])
        if i % 1000000 == 0:
            print i
        i += 1
    b = sparse.coo_matrix((values, (rows, columns)), shape=(row_num, column_num))
    return b.tocsr()


# read coo format matrix from file
def read_lil_mtx(filename):
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