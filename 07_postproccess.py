import numpy as np
import pandas as pd

aaa = pd.read_csv("./output/lightgbm_raw.log", names=["A"]).values

lgb_raw_logloss = np.concatenate([np.array(aaa[i][0].split())[3::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_logloss = pd.DataFrame(lgb_raw_logloss)
lgb_raw_logloss.columns = ['train_logloss', 'valid_logloss', 'test_logloss', 'test_mid_0_logloss',
       'test_mid_1_logloss', 'test_mid_2_logloss', 'test_mid_3_logloss',
       'test_mid_4_logloss', 'test_mid_5_logloss', 'test_mid_6_logloss',
       'test_mid_7_logloss', 'test_mid_8_logloss', 'test_mid_9_logloss']
lgb_raw_logloss.to_csv("./output/lightgbm_raw_logloss.csv", index=False)

lgb_raw_auc = np.concatenate([np.array(aaa[i][0].split())[6::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_auc = pd.DataFrame(lgb_raw_auc)
lgb_raw_auc.columns = ['train_auc', 'valid_auc', 'test_auc', 'test_mid_0_auc',
       'test_mid_1_auc', 'test_mid_2_auc', 'test_mid_3_auc', 'test_mid_4_auc',
       'test_mid_5_auc', 'test_mid_6_auc', 'test_mid_7_auc', 'test_mid_8_auc',
       'test_mid_9_auc']
lgb_raw_auc.to_csv("./output/lightgbm_raw_auc.csv", index=False)

lgb_raw_acc = np.concatenate([np.array(aaa[i][0].split())[9::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_acc = pd.DataFrame(1 - lgb_raw_acc)
lgb_raw_acc.columns = ['train_acc', 'valid_acc', 'test_acc', 'test_mid_0_acc',
       'test_mid_1_acc', 'test_mid_2_acc', 'test_mid_3_acc', 'test_mid_4_acc',
       'test_mid_5_acc', 'test_mid_6_acc', 'test_mid_7_acc', 'test_mid_8_acc',
       'test_mid_9_acc']
lgb_raw_acc.to_csv("./output/lightgbm_raw_acc.csv", index=False)

aaa = pd.read_csv("./output/lightgbm_std.log", names=["A"]).values

lgb_raw_logloss = np.concatenate([np.array(aaa[i][0].split())[3::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_logloss = pd.DataFrame(lgb_raw_logloss)
lgb_raw_logloss.columns = ['train_logloss', 'valid_logloss', 'test_logloss', 'test_mid_0_logloss',
       'test_mid_1_logloss', 'test_mid_2_logloss', 'test_mid_3_logloss',
       'test_mid_4_logloss', 'test_mid_5_logloss', 'test_mid_6_logloss',
       'test_mid_7_logloss', 'test_mid_8_logloss', 'test_mid_9_logloss']
lgb_raw_logloss.to_csv("./output/lightgbm_std_logloss.csv", index=False)

lgb_raw_auc = np.concatenate([np.array(aaa[i][0].split())[6::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_auc = pd.DataFrame(lgb_raw_auc)
lgb_raw_auc.columns = ['train_auc', 'valid_auc', 'test_auc', 'test_mid_0_auc',
       'test_mid_1_auc', 'test_mid_2_auc', 'test_mid_3_auc', 'test_mid_4_auc',
       'test_mid_5_auc', 'test_mid_6_auc', 'test_mid_7_auc', 'test_mid_8_auc',
       'test_mid_9_auc']
lgb_raw_auc.to_csv("./output/lightgbm_std_auc.csv", index=False)

lgb_raw_acc = np.concatenate([np.array(aaa[i][0].split())[9::9].astype("float").reshape(1, -1) for i in range(aaa.shape[0])], axis=0)
lgb_raw_acc = pd.DataFrame(1 - lgb_raw_acc)
lgb_raw_acc.columns = ['train_acc', 'valid_acc', 'test_acc', 'test_mid_0_acc',
       'test_mid_1_acc', 'test_mid_2_acc', 'test_mid_3_acc', 'test_mid_4_acc',
       'test_mid_5_acc', 'test_mid_6_acc', 'test_mid_7_acc', 'test_mid_8_acc',
       'test_mid_9_acc']
lgb_raw_acc.to_csv("./output/lightgbm_std_acc.csv", index=False)