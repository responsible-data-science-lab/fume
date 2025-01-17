"""
Data utility methods to make life easier.
"""
import os

import numpy as np


def get_data(dataset, data_dir='data', continuous=True):
    """
    Returns a train and test set from the desired dataset.
    """
    in_dir = os.path.join(data_dir, dataset)

    if continuous:
        in_dir = os.path.join(in_dir, 'continuous')

    assert os.path.exists(in_dir)

    train = np.load(os.path.join(in_dir, 'train.npy')).astype(np.float32)
    test = np.load(os.path.join(in_dir, 'test.npy')).astype(np.float32)

    if not continuous:
        assert np.all(np.unique(train) == np.array([0, 1]))
        assert np.all(np.unique(test) == np.array([0, 1]))

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test
