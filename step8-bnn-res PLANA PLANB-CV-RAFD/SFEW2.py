# -*- coding:utf-8 -*-


import numpy as np
import h5py


FEAT_TYPE = 'RAW-F583'


def _load_raw(matpath):
    print('load ' + matpath)

    h5 = h5py.File(matpath, 'r')
    x = h5['F'][:]
    # x = np.reshape(x, [x.shape[0], x.shape[1] * x.shape[2]])
    # x = np.float32(x)
    x = np.int8(x)
    x *= 2
    x -= 1

    y = h5['LBL'][:]
    if np.any(y == 0):# patch for unlabeled lfw, all LBL==0
        y += 1
    # y = np.int32(y) - 1
    y = np.int8(y) - 1
    y = np.hstack(y)
    y = np.float32(np.eye(7)[y])  # lbl -> onehot
    y *= 2
    y -= 1

    return [x, y]


def load_train_val():
    # 加载数据
    [train_x, train_y] = _load_raw('../DATA-CROP-' + FEAT_TYPE + '-LBL-RAFDTRAIN.mat')
    [val_x, val_y] = _load_raw('../DATA-CROP-' + FEAT_TYPE + '-LBL-RAFDVAL.mat')

    return [train_x, train_y, val_x, val_y]

def load_cv(fold):
    # 加载数据
    cv_x = []
    cv_y = []
    for i in xrange(1,6):
        [x, y] = _load_raw('../DATA-CROP-' + FEAT_TYPE + '-LBL-RAFDCV'+str(i)+'.mat')
        cv_x.append(x)
        cv_y.append(y)

    # 五折连接
    val_x = cv_x[fold-1]
    val_y = cv_y[fold-1]
    del cv_x[fold-1]
    del cv_y[fold-1]

    train_x = np.concatenate(cv_x)
    train_y = np.concatenate(cv_y)

    return [train_x, train_y, val_x, val_y]


def load_test():
    # 加载数据
    [test_x, test_y] = _load_raw('../DATA-CROP-' + FEAT_TYPE + '-LBL-RAFDTEST.mat')

    return [test_x, test_y]


def load_lfw():
    # 加载数据
    [lfw_x, _] = _load_raw('../DATA-CROP-' + FEAT_TYPE + '-LBL-LFW.mat')

    return [lfw_x]
