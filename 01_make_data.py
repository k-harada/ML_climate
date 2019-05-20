import numpy as np
import pandas as pd

# read dat file
data = pd.read_csv("./input/cru_ts4.01.1901.2016.tmp.dat", delim_whitespace=True, header=None).values

# reshape
data = data.reshape((-1, 360*720))

# ignore invalid grids
vvv = (data == -999).sum(axis=0)

# lat, lon
lat, lon = np.where(vvv.reshape((360, 720)) == 0)
lat = lat / 2 - 90
lon = (lon - 360) / 2

data_live = data[:, vvv == 0].transpose().astype("int32")
del data


np.random.seed(71)

train_size = 600000
valid_size = 200000
sample_size = train_size + valid_size

sample_indices = np.random.choice(67420*55, sample_size, replace=False)
# 0-54(start from 1901-1955, y:1931-1994)
X = np.zeros((sample_size, 360))
y10 = np.zeros((sample_size, 120))
y = np.zeros(sample_size).astype("int")
year = np.zeros(sample_size).astype("int")
longitude = np.zeros(sample_size)
latitude = np.zeros(sample_size)

for i in range(sample_size):
    sample_ind = sample_indices[i]
    grid_ind = sample_ind // 55
    year_ind = sample_ind % 55
    year[i] = year_ind + 1901
    longitude[i] = lon[grid_ind]
    latitude[i] = lat[grid_ind]
    X[i, :] = data_live[grid_ind, (year_ind*12):(year_ind*12+360)]
    y10[i, :] = data_live[grid_ind, (year_ind*12+360):(year_ind*12+480)]
    mean30 = X[i, :].mean()
    mean10 = y10[i, :].mean()
    if mean10 > mean30:
        y[i] = 1
    else:
        y[i] = 0


# 65-76(start from 1966-1977, y : 1996-2016)        
test_size = 200000

test_sample_indices = np.random.choice(67420*12, test_size, replace=False)

test_X = np.zeros((test_size, 360))
test_y10 = np.zeros((test_size, 120))
test_y = np.zeros(test_size).astype("int")
test_year = np.zeros(test_size).astype("int")
test_longitude = np.zeros(test_size)
test_latitude = np.zeros(test_size)

for i in range(test_size):
    sample_ind = test_sample_indices[i]
    grid_ind  = sample_ind // 12
    year_ind  = sample_ind % 12 + 65
    test_year[i] = year_ind + 1901
    test_longitude[i] = lon[grid_ind]
    test_latitude[i] = lat[grid_ind]
    test_X[i, :] = data_live[grid_ind, (year_ind*12):(year_ind*12+360)]
    test_y10[i, :] = data_live[grid_ind, (year_ind*12+360):(year_ind*12+480)]
    mean30 = test_X[i, :].mean()
    mean10 = test_y10[i, :].mean()
    if mean10 > mean30:
        test_y[i] = 1
    else:
        test_y[i] = 0
        

# 55-64(start from 1956-1965, y : 1986-2004)        
test_mid_size = 200000

test_mid_sample_indices = np.random.choice(67420, test_mid_size // 10, replace=False)

test_mid_X = np.zeros((test_mid_size, 360))
test_mid_y10 = np.zeros((test_mid_size, 120))
test_mid_y = np.zeros(test_mid_size).astype("int")
test_mid_year = np.zeros(test_mid_size).astype("int")
test_mid_longitude = np.zeros(test_mid_size)
test_mid_latitude = np.zeros(test_mid_size)

for year_ind in range(55, 65):
    for ind in range(test_mid_size // 10):
        grid_ind = test_mid_sample_indices[ind]
        i = (year_ind - 55)*(test_mid_size // 10) + ind
        test_mid_year[i] = year_ind + 1901
        test_mid_longitude[i] = lon[grid_ind]
        test_mid_latitude[i] = lat[grid_ind]
        test_mid_X[i, :] = data_live[grid_ind, (year_ind*12):(year_ind*12+360)]
        test_mid_y10[i, :] = data_live[grid_ind, (year_ind*12+360):(year_ind*12+480)]
        mean30 = test_mid_X[i, :].mean()
        mean10 = test_mid_y10[i, :].mean()
        if mean10 > mean30:
            test_mid_y[i] = 1
        else:
            test_mid_y[i] = 0


np.save("./data/train_X30.npy", X[:train_size, :])
np.save("./data/valid_X30.npy", X[train_size:, :])
np.save("./data/test_X30.npy", test_X)
np.save("./data/test_mid_X30.npy", test_mid_X)

np.save("./data/train_y10.npy", y10[:train_size, :])
np.save("./data/valid_y10.npy", y10[train_size:, :])
np.save("./data/test_y10.npy", test_y10)
np.save("./data/test_mid_y10.npy", test_mid_y10)

target_train = pd.DataFrame({
    "target" : y, 
    "start_year" : year, 
    "longitude" : longitude, 
    "latitude" : latitude
})
target_train.iloc[:train_size, :].to_csv("./data/train_base.csv", index=False)
target_train.iloc[train_size:, :].to_csv("./data/valid_base.csv", index=False)

target_test = pd.DataFrame({
    "target" : test_y, 
    "start_year" : test_year, 
    "longitude" : test_longitude, 
    "latitude" : test_latitude
})
target_test.to_csv("./data/test_base.csv", index=False)

target_test_mid = pd.DataFrame({
    "target" : test_mid_y, 
    "start_year" : test_mid_year, 
    "longitude" : test_mid_longitude, 
    "latitude" : test_mid_latitude
})
target_test_mid.to_csv("./data/test_mid_base.csv", index=False)
