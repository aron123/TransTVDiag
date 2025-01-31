import numpy as np
from sklearn.ensemble import IsolationForest

def k_sigma(train_arr, test_arr: np.array, k=3):
    mean = np.mean(train_arr)
    std = np.std(train_arr)
    up, lb=mean+k*std, mean-k*std
    ab_points=test_arr[(test_arr>up)|(test_arr<lb)]
    labels=np.array([0]*len(test_arr))
    ab_idxs=np.where((test_arr>up)|(test_arr<lb))
    labels[ab_idxs]=-1
    return ab_points, labels

def IsolationForest_detect(train_arr, test_arr):
    clf = IsolationForest(random_state=0, n_estimators=5)
    clf.fit(train_arr.reshape(-1,1))
    labels = clf.predict(test_arr.reshape(-1,1))
    ab_points = test_arr[labels==-1]
    return ab_points, labels
    