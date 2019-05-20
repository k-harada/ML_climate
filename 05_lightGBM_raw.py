import os
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import logging

from lightgbm.callback import _format_eval_result

np.random.seed(71)

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
sc = logging.StreamHandler()
logger.addHandler(sc)
fh = logging.FileHandler('./output/lightgbm_raw.log')
logger.addHandler(fh)

train_X = np.load("./data/train_X30.npy")
valid_X = np.load("./data/valid_X30.npy")
test_X = np.load("./data/test_X30.npy")
test_mid_X = np.load("./data/test_mid_X30.npy")

test_mid_size = test_mid_X.shape[0] // 10

train_y = pd.read_csv("./data/train_base.csv").target.values
valid_y = pd.read_csv("./data/valid_base.csv").target.values
test_y = pd.read_csv("./data/test_base.csv").target.values
test_mid_y = pd.read_csv("./data/test_mid_base.csv").target.values

train_xy = lgb.Dataset(train_X, train_y)
valid_xy = lgb.Dataset(valid_X, valid_y)
test_xy = lgb.Dataset(test_X, test_y)

test_mid_xy_0 = lgb.Dataset(test_mid_X[test_mid_size*0:test_mid_size*1, :], test_mid_y[test_mid_size*0:test_mid_size*1])
test_mid_xy_1 = lgb.Dataset(test_mid_X[test_mid_size*1:test_mid_size*2, :], test_mid_y[test_mid_size*1:test_mid_size*2])
test_mid_xy_2 = lgb.Dataset(test_mid_X[test_mid_size*2:test_mid_size*3, :], test_mid_y[test_mid_size*2:test_mid_size*3])
test_mid_xy_3 = lgb.Dataset(test_mid_X[test_mid_size*3:test_mid_size*4, :], test_mid_y[test_mid_size*3:test_mid_size*4])
test_mid_xy_4 = lgb.Dataset(test_mid_X[test_mid_size*4:test_mid_size*5, :], test_mid_y[test_mid_size*4:test_mid_size*5])
test_mid_xy_5 = lgb.Dataset(test_mid_X[test_mid_size*5:test_mid_size*6, :], test_mid_y[test_mid_size*5:test_mid_size*6])
test_mid_xy_6 = lgb.Dataset(test_mid_X[test_mid_size*6:test_mid_size*7, :], test_mid_y[test_mid_size*6:test_mid_size*7])
test_mid_xy_7 = lgb.Dataset(test_mid_X[test_mid_size*7:test_mid_size*8, :], test_mid_y[test_mid_size*7:test_mid_size*8])
test_mid_xy_8 = lgb.Dataset(test_mid_X[test_mid_size*8:test_mid_size*9, :], test_mid_y[test_mid_size*8:test_mid_size*9])
test_mid_xy_9 = lgb.Dataset(test_mid_X[test_mid_size*9:test_mid_size*10, :], test_mid_y[test_mid_size*9:test_mid_size*10])

params = {
    "objective" : "binary", 
    "metric" : ["binary_error", "binary_logloss", "auc"]
}
callbacks = [log_evaluation(logger, period=100)]

lgb.train(
    params, train_set=train_xy, valid_sets=[
        train_xy, valid_xy, test_xy, 
        test_mid_xy_0, test_mid_xy_1, test_mid_xy_2, test_mid_xy_3, 
        test_mid_xy_4, test_mid_xy_5, test_mid_xy_6, test_mid_xy_7, 
        test_mid_xy_8, test_mid_xy_9
    ], 
    num_boost_round=10000, verbose_eval=100, callbacks=callbacks
)