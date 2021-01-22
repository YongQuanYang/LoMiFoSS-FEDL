# 最优化求解不同权重的组合方案，但效果不佳

import argparse
import os
import random
import sys
import time
import data
import glob
import copy
import pickle
import numpy as np
from scipy import optimize
from sklearn.metrics import accuracy_score
parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--pred_path",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--label_path",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

def main():
    args = parser.parse_args()
    print("args:{}".format(args))
    pred_path = args.pred_path
    label_path = args.label_path

    pred_pkl_paths = glob.glob(pred_path)
    pred_list = []
    for pred_pkl_path in pred_pkl_paths:
        with open(pred_pkl_path, 'rb') as f:
            pkl = pickle.load(f, encoding='iso-8859-1')

        pred_list.append(pkl["logits"])

    with open(label_path, 'rb') as f:
        pkl = pickle.load(f, encoding='iso-8859-1')
    label = pkl["label"]

    def f(weights):
        # import pdb
        # pdb.set_trace()
        valid_preds = np.average(pred_list, axis=0, weights=weights)
        y_valid_pred_cls = np.argmax(valid_preds, axis=1)
        return y_valid_pred_cls

    def loss_function(weights):
        y_valid_pred_cls = f(weights)
        acc = accuracy_score(label, y_valid_pred_cls)
        return 1 - acc
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
    # opt_weights = optimize.minimize(loss_function,
    #                                 [1 / len(pred_list)] * len(pred_list),
    #                                 constraints=({'type': 'eq', 'fun': lambda w: 1 - sum(w)}),
    #                                 method='Nelder-Mead',  # 'SLSQP',
    #                                 bounds=[(0.0, 1.0)] * len(pred_list),
    #                                 options={'fatol': 1e-4, 'disp': True, 'maxiter': 300},
    #                                 )['x']

    opt_weights = [1 / len(pred_list)] * len(pred_list)
    print('Optimum weights = ', opt_weights, 'with loss', loss_function(opt_weights))

    def acc_function(weights):
        y_valid_pred_cls = f(weights)
        acc = accuracy_score(label, y_valid_pred_cls)
        return acc

    print('Ensembled Accuracy =', acc_function(opt_weights))

    # # double check answers
    # n = 5
    # y_valid_pred_cls = f(opt_weights)
    # for result, ref in zip(y_valid_pred_cls[:n], ref_valid_cls[:n]):
    #     print(result, '\t', ref)

if __name__ == '__main__':
    main()