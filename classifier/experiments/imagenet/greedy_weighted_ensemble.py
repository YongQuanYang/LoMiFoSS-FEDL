# 贪婪算法求解不同权重组合方案
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
parser.add_argument('--ensemble_mode', type=str, default="greedy" ,choices=['greedy', 'select', "random_greedy"])

parser.add_argument('--select_index', default=[], type=int, nargs="+")
parser.add_argument('--random_times', default=100, type=int)

def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return averaged_model_parameter + \
           (model_parameter - averaged_model_parameter) / (num_averaged + 1)

def greedy_ensemble(metric_np_index, pred_list, label):
    bast_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    for i in range(len(metric_np_index)):
        avg_logit = avg_fn(ensemble_logit, pred_list[metric_np_index[i]], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print("i:{}, metric_np_index[i]:{} avg_acc:{}, bast_acc:{}， num_averaged:{}".format(i, metric_np_index[i], avg_acc, bast_acc, num_averaged))
        if avg_acc > bast_acc:
            ensemble_list.append(metric_np_index[i])
            bast_acc = avg_acc
            ensemble_logit = avg_logit
            num_averaged += 1
    print("best acc:{}, ensemble_list:{}".format(bast_acc, ensemble_list))


def random_greedy_ensemble(metric_np_index, pred_list, label, random_times=100):
    bast_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0

    for i in range(random_times):
        if i == 0:
            index = 0
        else:
            index = random.choice(metric_np_index)

        avg_logit = avg_fn(ensemble_logit, pred_list[metric_np_index[index]], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print("i:{}, metric_np_index[i]:{} avg_acc:{}, bast_acc:{}， num_averaged:{}".format(index, metric_np_index[index], avg_acc, bast_acc, num_averaged))
        if avg_acc > bast_acc:
            ensemble_list.append(metric_np_index[index])
            bast_acc = avg_acc
            ensemble_logit = avg_logit
            num_averaged += 1
    print("best acc:{}, ensemble_list:{}".format(bast_acc, ensemble_list))


def select_ensemble(select_np_index, pred_list, label):
    bast_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    for i in range(len(select_np_index)):
        avg_logit = avg_fn(ensemble_logit, pred_list[select_np_index[i]], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print("i:{}, select_np_index[i]:{} avg_acc:{}, bast_acc:{}， num_averaged:{}".format(i, select_np_index[i], avg_acc, bast_acc, num_averaged))
        ensemble_list.append(select_np_index[i])
        bast_acc = avg_acc
        ensemble_logit = avg_logit
        num_averaged += 1
    print("best acc:{}, ensemble_list:{}".format(bast_acc, ensemble_list))

def get_metric(logit, label):
    y_valid_pred_cls = np.argmax(logit, axis=1)
    acc = accuracy_score(label, y_valid_pred_cls)
    return acc


def main():
    args = parser.parse_args()
    print("args:{}".format(args))
    pred_path = args.pred_path
    label_path = args.label_path
    ensemble_mode = args.ensemble_mode
    select_index = args.select_index
    random_times = args.random_times

    pred_pkl_paths = glob.glob(pred_path)
    pred_pkl_paths = sorted(pred_pkl_paths)
    pred_list = []
    pred_pkl_paths_sort = []
    for i in range(len(pred_pkl_paths)):
        pred_pkl_path = pred_path.replace("*", str(i))
        pred_pkl_paths_sort.append(pred_pkl_path)
        with open(pred_pkl_path, 'rb') as f:
            pkl = pickle.load(f, encoding='iso-8859-1')

        pred_list.append(pkl["logits"])
    print("pred_pkl_paths_sort:{}".format(pred_pkl_paths_sort))

    with open(label_path, 'rb') as f:
        pkl = pickle.load(f, encoding='iso-8859-1')
    label = pkl["label"]

    if ensemble_mode == "greedy":
        metric_list = []
        for i, logit in enumerate(pred_list):
            acc = get_metric(logit, label)
            metric_list.append(acc)
        print("metric_list:{}".format(metric_list))

        metric_np = np.array(metric_list)
        # 降序
        metric_np_index = np.argsort(-metric_np)
        # 顺序
        #metric_np_index = np.array(list(range(len(pred_pkl_paths))))
        print("sort metric_list index:{}".format(metric_np_index))
        greedy_ensemble(metric_np_index, pred_list, label)
    elif ensemble_mode == "select":
        select_ensemble(select_index, pred_list, label)
    elif ensemble_mode == "random_greedy":
        metric_list = []
        for i, logit in enumerate(pred_list):
            acc = get_metric(logit, label)
            metric_list.append(acc)
        print("metric_list:{}".format(metric_list))

        metric_np = np.array(metric_list)
        # 降序
        metric_np_index = np.argsort(-metric_np)
        # 顺序
        # metric_np_index = np.array(list(range(len(pred_pkl_paths))))
        print("sort metric_list index:{}".format(metric_np_index))
        random_greedy_ensemble(metric_np_index, pred_list, label, random_times=random_times)
    else:
        raise NotImplementedError




if __name__ == '__main__':
    main()