import os
import numpy as np
import pandas as pd
from  tqdm import tqdm
import tensorflow as tf

epochs = 300

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(71)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(71)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import Adam

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

def standardize(X):
    X_st = X - X.min(axis=1, keepdims=True)
    X_st = X_st / X_st.max(axis=1, keepdims=True)
    X_re = X_st.reshape((-1, 12, 30, 1))
    return X_re

train_X = standardize(train_X)
valid_X = standardize(valid_X)
test_X = standardize(test_X)
test_mid_X = standardize(test_mid_X)

def scores(y, pred):
    acc = accuracy_score(y, pred > 0.5)
    logloss = log_loss(y, np.minimum(np.maximum(pred.astype("float64"), 10**(-15)), 1 - 10**(-15)))
    auc = roc_auc_score(y, pred)
    return acc, logloss, auc

# LeNet-like model (small)
model_small = Sequential()
model_small.add(Conv2D(6, (12, 5), activation="relu", input_shape=(12, 30, 1)))
model_small.add(MaxPool2D((1, 2)))
model_small.add(Conv2D(16, (1, 5), activation="relu"))
model_small.add(MaxPool2D((1, 2)))
model_small.add(Flatten())
model_small.add(Dense(120, activation="relu"))
model_small.add(Dense(1, activation="sigmoid"))
model_small.compile(loss="binary_crossentropy", optimizer="Adam")


def fit_climate(model, epochs):
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

    for e in tqdm(range(epochs)):
        _ = model.fit(train_X, train_y, batch_size=128, epochs=1, shuffle=False, verbose=False)

        pred_train = model.predict(train_X)
        pred_valid = model.predict(valid_X)
        pred_test = model.predict(test_X)
        pred_test_mid = model.predict(test_mid_X)

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

    # collect data
    # acc
    acc_df = pd.DataFrame({
        "train_acc":train_acc, 
        "valid_acc":valid_acc, 
        "test_acc":test_acc
    })
    for j in range(10):
        acc_df["test_mid_"+str(j)+"_acc"] = test_mid_acc[j]

    # logloss
    logloss_df = pd.DataFrame({
        "train_logloss":train_logloss, 
        "valid_logloss":valid_logloss, 
        "test_logloss":test_logloss
    })
    for j in range(10):
        logloss_df["test_mid_"+str(j)+"_logloss"] = test_mid_logloss[j]

    # auc
    auc_df = pd.DataFrame({
        "train_auc":train_auc, 
        "valid_auc":valid_auc, 
        "test_auc":test_auc
    })
    for j in range(10):
        auc_df["test_mid_"+str(j)+"_auc"] = test_mid_auc[j]

    return acc_df, logloss_df, auc_df


# run and write
acc_df, logloss_df, auc_df = fit_climate(model_small, epochs)
acc_df.to_csv("./output/nn_small_acc.csv", index=False)
logloss_df.to_csv("./output/nn_small_logloss.csv", index=False)
auc_df.to_csv("./output/nn_small_auc.csv", index=False)
