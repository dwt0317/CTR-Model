# -*- coding:utf-8 -*-


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