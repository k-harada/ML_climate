import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_X = np.load("./data/train_X30.npy")
valid_X = np.load("./data/valid_X30.npy")
test_X = np.load("./data/test_X30.npy")
test_mid_X = np.load("./data/test_mid_X30.npy")

test_mid_size = test_mid_X.shape[0] // 10

train_y = pd.read_csv("./data/train_base.csv").target.values
valid_y = pd.read_csv("./data/valid_base.csv").target.values
test_y = pd.read_csv("./data/test_base.csv").target.values
test_mid_y = pd.read_csv("./data/test_mid_base.csv").target.values

def standardize_lr(X):
    X_st = X - X.min(axis=1, keepdims=True)
    X_st = X_st / X_st.max(axis=1, keepdims=True)
    return X_st

train_X_st = standardize_lr(train_X)
valid_X_st = standardize_lr(valid_X)
test_X_st = standardize_lr(test_X)
test_mid_X_st = standardize_lr(test_mid_X)

train_X2 = (train_X_st[:, 180:].mean(axis=1) - train_X_st[:, :180].mean(axis=1)).reshape(-1, 1)
valid_X2 = (valid_X_st[:, 180:].mean(axis=1) - valid_X_st[:, :180].mean(axis=1)).reshape(-1, 1)
test_X2 = (test_X_st[:, 180:].mean(axis=1) - test_X_st[:, :180].mean(axis=1)).reshape(-1, 1)
test_mid_X2 = (test_mid_X_st[:, 180:].mean(axis=1) - test_mid_X_st[:, :180].mean(axis=1)).reshape(-1, 1)

def scores(y, pred):
    acc = accuracy_score(y, pred > 0.5)
    logloss = log_loss(y, pred)
    auc = roc_auc_score(y, pred)
    return acc, logloss, auc

model_raw = LogisticRegression(random_state=71)
model_std = LogisticRegression(random_state=71)
model_std_2 = LogisticRegression(random_state=71)

pred_train_list = []
pred_valid_list = []
pred_test_list = []
pred_test_mid_list = []

print("Raw Logistic Regression")
_ = model_raw.fit(train_X, train_y)
pred_train = model_raw.predict_proba(train_X)[:, 1]
pred_valid = model_raw.predict_proba(valid_X)[:, 1]
pred_test = model_raw.predict_proba(test_X)[:, 1]
pred_test_mid = model_raw.predict_proba(test_mid_X)[:, 1]
pred_train_list.append(pred_train)
pred_valid_list.append(pred_valid)
pred_test_list.append(pred_test)
pred_test_mid_list.append(pred_test_mid)

print("Standardized Logistic Regression")
_ = model_std.fit(train_X_st, train_y)
pred_train = model_std.predict_proba(train_X_st)[:, 1]
pred_valid = model_std.predict_proba(valid_X_st)[:, 1]
pred_test = model_std.predict_proba(test_X_st)[:, 1]
pred_test_mid = model_std.predict_proba(test_mid_X_st)[:, 1]
pred_train_list.append(pred_train)
pred_valid_list.append(pred_valid)
pred_test_list.append(pred_test)
pred_test_mid_list.append(pred_test_mid)

print("One feature Logistic Regression")
_ = model_std_2.fit(train_X2, train_y)
pred_train = model_std_2.predict_proba(train_X2)[:, 1]
pred_valid = model_std_2.predict_proba(valid_X2)[:, 1]
pred_test = model_std_2.predict_proba(test_X2)[:, 1]
pred_test_mid = model_std_2.predict_proba(test_mid_X2)[:, 1]
pred_train_list.append(pred_train)
pred_valid_list.append(pred_valid)
pred_test_list.append(pred_test)
pred_test_mid_list.append(pred_test_mid)

print("Collecting results")
train_acc = []
train_logloss = []
train_auc = []
valid_acc = []
valid_logloss = []
valid_auc = []
test_acc = []
test_logloss = []
test_auc = []
test_mid_acc = [[] for _ in range(10)]
test_mid_logloss = [[] for _ in range(10)]
test_mid_auc = [[] for _ in range(10)]


for i in range(3):
    pred_train = pred_train_list[i]
    pred_valid = pred_valid_list[i]
    pred_test = pred_test_list[i]
    pred_test_mid = pred_test_mid_list[i]
    
    acc, logloss, auc = scores(train_y, pred_train)
    train_acc.append(acc)
    train_logloss.append(logloss)
    train_auc.append(auc)

    acc, logloss, auc = scores(valid_y, pred_valid)
    valid_acc.append(acc)
    valid_logloss.append(logloss)
    valid_auc.append(auc)

    acc, logloss, auc = scores(test_y, pred_test)
    test_acc.append(acc)
    test_logloss.append(logloss)
    test_auc.append(auc)

    for j in range(10):
        acc, logloss, auc = scores(test_mid_y[j*test_mid_size:(j+1)*test_mid_size], pred_test_mid[j*test_mid_size:(j+1)*test_mid_size])
        test_mid_acc[j].append(acc)
        test_mid_logloss[j].append(logloss)
        test_mid_auc[j].append(auc)

print("Write csv")
acc_df = pd.DataFrame({
    "train_acc":train_acc, 
    "valid_acc":valid_acc, 
    "test_acc":test_acc
})
for j in range(10):
    acc_df["test_mid_"+str(j)+"_acc"] = test_mid_acc[j]

logloss_df = pd.DataFrame({
    "train_logloss":train_logloss, 
    "valid_logloss":valid_logloss, 
    "test_logloss":test_logloss
})
for j in range(10):
    logloss_df["test_mid_"+str(j)+"_logloss"] = test_mid_logloss[j]
    
auc_df = pd.DataFrame({
    "train_auc":train_auc, 
    "valid_auc":valid_auc, 
    "test_auc":test_auc
})
for j in range(10):
    auc_df["test_mid_"+str(j)+"_auc"] = test_mid_auc[j]
    
acc_df.to_csv("./output/logistic_acc.csv", index=False)
logloss_df.to_csv("./output/logistic_logloss.csv", index=False)
auc_df.to_csv("./output/logistic_auc.csv", index=False)
print("Done")