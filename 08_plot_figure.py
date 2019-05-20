import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logloss_df = pd.read_csv("./output/nn_small_logloss.csv")
acc_df = pd.read_csv("./output/nn_small_acc.csv")
auc_df = pd.read_csv("./output/nn_small_auc.csv")

plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 

fig, (logloss, acc, auc) = plt.subplots(ncols=3, figsize=(15,4))

logloss.plot(logloss_df["train_logloss"], label="train_logloss")
logloss.plot(logloss_df["valid_logloss"], label="valid_logloss")
logloss.plot(logloss_df["test_logloss"], label="test_logloss")
logloss.set_xlabel("epochs")
logloss.set_ylabel("log loss")
logloss.legend()

acc.plot(acc_df["train_acc"], label="train_acc")
acc.plot(acc_df["valid_acc"], label="valid_acc")
acc.plot(acc_df["test_acc"], label="test_acc")
acc.set_xlabel("epochs")
acc.set_ylabel("accuracy")
acc.legend()

auc.plot(auc_df["train_auc"], label="train_auc")
auc.plot(auc_df["valid_auc"], label="valid_auc")
auc.plot(auc_df["test_auc"], label="test_auc")
auc.set_xlabel("epochs")
auc.set_ylabel("AUC")
auc.legend()

fig.savefig('./figures/nn_small.png')


logloss_df = pd.read_csv("./output/nn_large_logloss.csv")
acc_df = pd.read_csv("./output/nn_large_acc.csv")
auc_df = pd.read_csv("./output/nn_large_auc.csv")

plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 

fig, (logloss, acc, auc) = plt.subplots(ncols=3, figsize=(15,4))

logloss.plot(logloss_df["train_logloss"], label="train_logloss")
logloss.plot(logloss_df["valid_logloss"], label="valid_logloss")
logloss.plot(logloss_df["test_logloss"], label="test_logloss")
logloss.set_xlabel("epochs")
logloss.set_ylabel("log loss")
logloss.legend()

acc.plot(acc_df["train_acc"], label="train_acc")
acc.plot(acc_df["valid_acc"], label="valid_acc")
acc.plot(acc_df["test_acc"], label="test_acc")
acc.set_xlabel("epochs")
acc.set_ylabel("accuracy")
acc.legend()

auc.plot(auc_df["train_auc"], label="train_auc")
auc.plot(auc_df["valid_auc"], label="valid_auc")
auc.plot(auc_df["test_auc"], label="test_auc")
auc.set_xlabel("epochs")
auc.set_ylabel("AUC")
auc.legend()

fig.savefig('./figures/nn_large.png')


logloss_df = pd.read_csv("./output/lightgbm_raw_logloss.csv")
acc_df = pd.read_csv("./output/lightgbm_raw_acc.csv")
auc_df = pd.read_csv("./output/lightgbm_raw_auc.csv")

plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 

fig, (logloss, acc, auc) = plt.subplots(ncols=3, figsize=(15,4))

logloss.plot(logloss_df["train_logloss"], label="train_logloss")
logloss.plot(logloss_df["valid_logloss"], label="valid_logloss")
logloss.plot(logloss_df["test_logloss"], label="test_logloss")
logloss.set_xlabel("nrounds / 100")
logloss.set_ylabel("log loss")
logloss.legend()

acc.plot(acc_df["train_acc"], label="train_acc")
acc.plot(acc_df["valid_acc"], label="valid_acc")
acc.plot(acc_df["test_acc"], label="test_acc")
acc.set_xlabel("nrounds / 100")
acc.set_ylabel("accuracy")
acc.legend()

auc.plot(auc_df["train_auc"], label="train_auc")
auc.plot(auc_df["valid_auc"], label="valid_auc")
auc.plot(auc_df["test_auc"], label="test_auc")
auc.set_xlabel("nrounds / 100")
auc.set_ylabel("AUC")
auc.legend()
fig.savefig('./figures/lightgbm_raw.png')


logloss_df = pd.read_csv("./output/lightgbm_std_logloss.csv")
acc_df = pd.read_csv("./output/lightgbm_std_acc.csv")
auc_df = pd.read_csv("./output/lightgbm_std_auc.csv")

plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 

fig, (logloss, acc, auc) = plt.subplots(ncols=3, figsize=(15,4))

logloss.plot(logloss_df["train_logloss"], label="train_logloss")
logloss.plot(logloss_df["valid_logloss"], label="valid_logloss")
logloss.plot(logloss_df["test_logloss"], label="test_logloss")
logloss.set_xlabel("nrounds / 100")
logloss.set_ylabel("log loss")
logloss.legend()

acc.plot(acc_df["train_acc"], label="train_acc")
acc.plot(acc_df["valid_acc"], label="valid_acc")
acc.plot(acc_df["test_acc"], label="test_acc")
acc.set_xlabel("nrounds / 100")
acc.set_ylabel("accuracy")
acc.legend()

auc.plot(auc_df["train_auc"], label="train_auc")
auc.plot(auc_df["valid_auc"], label="valid_auc")
auc.plot(auc_df["test_auc"], label="test_auc")
auc.set_xlabel("nrounds / 100")
auc.set_ylabel("AUC")
auc.legend()

fig.savefig('./figures/lightgbm_std.png')