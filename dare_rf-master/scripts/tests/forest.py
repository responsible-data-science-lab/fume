import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dare
from utility import data_util

def load_data(dataset, data_dir):

    if dataset == 'iris':
        data = load_iris()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        indices = np.where(y != 2)[0]
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = X, X, y, y

    elif dataset == 'boston':
        data = load_boston()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

        X_train, X_test, y_train, y_test = X, X, y, y

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(dataset, data_dir)

        X_train = X_train[:,:50]
        X_test = X_test[:,:50]

    return X_train, X_test, y_train, y_test


def main(args):

    # get data
    X_train, X_test, y_train, y_test = load_data(args.dataset, args.data_dir)

    # train
    topd = 0
    k = 100
    n_estimators = 100
    max_depth = 20
    seed = 1
    n_delete = 100

    if args.model == 'dare':
        model = dare.Forest(topd=topd, k=k, n_estimators=n_estimators,
                            max_depth=max_depth, random_state=seed)

    elif args.model == 'sklearn':
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=seed)

    start = time.time()
    model = dare.Forest(topd=topd, k=k, n_estimators=n_estimators,
                        max_depth=max_depth, random_state=seed)
    model = model.fit(X_train, y_train)
    train_time = time.time() - start
    print('train time: {:.3f}s'.format(train_time))

    # predict
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print('ACC: {:.3f}, AUC: {:.3f}'.format(acc, auc))

    # delete training data
    cum_delete_time = 0
    if args.delete and not args.simulate:
        delete_indices = np.random.default_rng(seed=seed).choice(X_train.shape[0], size=n_delete, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        # delete each sample
        for delete_ndx in delete_indices:
            start = time.time()
            model.delete(delete_ndx)
            delete_time = time.time() - start
            cum_delete_time += delete_time
            print('\ndeleted instance, {}: {:.3f}s'.format(delete_ndx, delete_time))

        types, depths, costs = model.get_delete_metrics()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))
        print('costs: {}'.format(costs))

        avg_delete_time = cum_delete_time / len(delete_indices)
        print('train time: {:.3f}s'.format(train_time))
        print('avg. delete time: {:.3f}s'.format(avg_delete_time))

    # simulate the deletion of each instance
    elif args.delete and args.simulate:
        delete_indices = np.random.default_rng(seed=seed).choice(X_train.shape[0], size=n_delete, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        # cumulative time
        cum_delete_time = 0
        cum_sim_time = 0

        # simulate and delete each sample
        for delete_ndx in delete_indices:

            # simulate the deletion
            start = time.time()
            n_samples_to_retrain = model.sim_delete(delete_ndx)
            if args.test_idempotency:
                n_samples_to_retrain = model.sim_delete(delete_ndx)
            sim_time = time.time() - start
            cum_sim_time += sim_time
            print('\nsimulated instance, {}: {:.3f}s, no. samples: {:,}'.format(
                  delete_ndx, sim_time, n_samples_to_retrain))

            # delete
            start = time.time()
            model.delete(delete_ndx)
            delete_time = time.time() - start
            cum_delete_time += delete_time
            print('deleted instance, {}: {:.3f}s'.format(delete_ndx, delete_time))

        types, depths, costs = model.get_delete_metrics()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))
        print('costs: {}'.format(costs.shape))

        avg_sim_time = cum_sim_time / len(delete_indices)
        avg_delete_time = cum_delete_time / len(delete_indices)

        print('avg. sim. time: {:.5f}s'.format(avg_sim_time))
        print('avg. delete time: {:.5f}s'.format(avg_delete_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    parser.add_argument('--model', type=str, default='dare', help='dare or sklearn')
    parser.add_argument('--delete', action='store_true', help='whether to deletion or not.')
    parser.add_argument('--simulate', action='store_true', help='whether to simulate deletions or not.')
    parser.add_argument('--test_idempotency', action='store_true', help='simulate deletion multiple times.')
    args = parser.parse_args()
    main(args)
