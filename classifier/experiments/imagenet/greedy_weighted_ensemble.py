"""

Authors: lvhaijun01@baidu.com
Date:     2021-02-03 14:11
"""

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
parser.add_argument('--ensemble_mode', type=str, default="greedy" ,choices=['greedy', "greedy_brute_force", 'select', "random_greedy", "random_greedy_v2", "swa_all_ensemble", "avg_step_all_ensemble"])

parser.add_argument('--select_index', default=[], type=int, nargs="+")
parser.add_argument('--single_random_times', default=100, type=int)
parser.add_argument('--put_back', default=1, type=int)
parser.add_argument('--experi_times', default=10, type=int)

def swa_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return averaged_model_parameter + \
           (model_parameter - averaged_model_parameter) / (num_averaged + 1)
def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return (averaged_model_parameter + model_parameter) / 2


def greedy_brute_force_ensemble(metric_np_index, pred_list, label):
    """
    ensemble ckpt one by one from best val acc to worst val acc, if result is better, then reserve ckpt, otherwise stop.
    Args:
        metric_np_index:
        pred_list:
        label:

    Returns:

    """
    best_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    single_metric_index = list(copy.deepcopy(metric_np_index))
    for i in range(len(single_metric_index)):
        single_best_acc = best_acc
        single_best_index = -1
        single_best_logit = 0
        for j in range(i, len(single_metric_index)):
            if j in ensemble_list:
                continue
            avg_logit = avg_fn(ensemble_logit, pred_list[single_metric_index[j]], num_averaged)
            avg_acc = get_metric(avg_logit, label)
            print("i:{}, j:{}, single_metric_index:{} avg_acc:{}, single_best_acc:{}，single_best_index:{}, num_averaged:{}, best_acc:{}".format(i, j, single_metric_index[j], avg_acc, single_best_acc, single_best_index, num_averaged, best_acc))
            if avg_acc > single_best_acc:
                single_best_acc = avg_acc
                single_best_logit = avg_logit
                single_best_index = j

        print("i:{}, single_best_acc:{}，best_acc:{}, single_best_index:{} num_averaged:{}".format(i,  single_best_acc, best_acc, single_best_index, num_averaged))
        if single_best_acc > best_acc:
            ensemble_list.append(single_metric_index[single_best_index])
            best_acc = single_best_acc
            ensemble_logit = single_best_logit
            num_averaged += 1
        else:
            break
    print("best acc:{}, ensemble_list:{}".format(best_acc, ensemble_list))


def greedy_ensemble(metric_np_index, pred_list, label):
    """
    ensemble ckpt one by one from best val acc to worst val acc, if result is better, then reserve ckpt, otherwise throw away ckpt.
    Args:
        metric_np_index:
        pred_list:
        label:

    Returns:

    """
    best_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    for i in range(len(metric_np_index)):
        avg_logit = avg_fn(ensemble_logit, pred_list[metric_np_index[i]], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print("i:{}, metric_np_index[i]:{} avg_acc:{}, best_acc:{}， num_averaged:{}".format(i, metric_np_index[i], avg_acc, best_acc, num_averaged))
        if avg_acc > best_acc:
            ensemble_list.append(metric_np_index[i])
            best_acc = avg_acc
            ensemble_logit = avg_logit
            num_averaged += 1
    print("best acc:{}, ensemble_list:{}".format(best_acc, ensemble_list))



def single_random_greedy_ensemble(metric_np_index, pred_list, label, random_times=100, put_back=True):
    """

    Args:
        metric_np_index:
        pred_list:
        label:
        random_times:
        put_back: 用后放回

    Returns:

    """
    best_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    single_metric_index = list(copy.deepcopy(metric_np_index))
    for i in range(random_times):
        if i == 0:
            index = 0
        else:
            index = random.choice(single_metric_index)

        avg_logit = avg_fn(ensemble_logit, pred_list[index], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print(
            "i:{}, single_metric_index:{} avg_acc:{}, best_acc:{}， num_averaged:{}".format(i, index,
                                                                                          avg_acc, best_acc,
                                                                                          num_averaged))
        if avg_acc > best_acc:
            ensemble_list.append(index)
            best_acc = avg_acc
            ensemble_logit = avg_logit
            num_averaged += 1
            if not put_back:
                single_metric_index.remove(index)

    print("best acc:{}, ensemble_list:{}".format(best_acc, ensemble_list))
    return best_acc, ensemble_list

def random_greedy_ensemble(metric_np_index, pred_list, label, random_times=100, experi_times=10, put_back=True):
    """
    do random greedy ckpt experiment random_times, each experiment contains experi_times ensemble behaviors,
    each ensemble behavior ensemble better result with ckpt put_back or not.
    Args:
        metric_np_index:
        pred_list:
        label:
        random_times:
        experi_times:
        put_back:

    Returns:

    """
    best_acc = 0
    ensemble_list = []
    best_time = 0
    for i in range(experi_times):
        single_exper_acc, single_ensemble_list = single_random_greedy_ensemble(metric_np_index, pred_list, label, random_times, put_back=put_back)
        if single_exper_acc > best_acc:
            best_acc = single_exper_acc
            ensemble_list = single_ensemble_list
            best_time = i
    print("best acc:{}, ensemble_list:{}, in {} time".format(best_acc, ensemble_list, best_time))


def single_random_greedy_v2_ensemble(metric_np_index, pred_list, label):
    """

    Args:
        metric_np_index:
        pred_list:
        label:

    Returns:

    """
    best_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    single_metric_index = list(copy.deepcopy(metric_np_index))
    for i in range(len(single_metric_index)):
        if i == 0:
            index = 0
        else:
            index = random.choice(single_metric_index)

        avg_logit = avg_fn(ensemble_logit, pred_list[index], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print(
            "i:{}, single_metric_index:{} avg_acc:{}, best_acc:{}， num_averaged:{}".format(i, index,
                                                                                          avg_acc, best_acc,
                                                                                          num_averaged))
        if avg_acc > best_acc:
            ensemble_list.append(index)
            best_acc = avg_acc
            ensemble_logit = avg_logit
            num_averaged += 1
        single_metric_index.remove(index)

    print("best acc:{}, ensemble_list:{}".format(best_acc, ensemble_list))
    return best_acc, ensemble_list


def random_greedy_v2_ensemble(metric_np_index, pred_list, label, experi_times=10):
    """
    进行随机不放回集成，最多n次
    Args:
        metric_np_index:
        pred_list:
        label:
        experi_times:

    Returns:

    """
    best_acc = 0
    ensemble_list = []
    best_time = 0
    for i in range(experi_times):
        single_exper_acc, single_ensemble_list = single_random_greedy_v2_ensemble(metric_np_index, pred_list, label)
        if single_exper_acc > best_acc:
            best_acc = single_exper_acc
            ensemble_list = single_ensemble_list
            best_time = i
    print("best acc:{}, ensemble_list:{}, in {} time".format(best_acc, ensemble_list, best_time))

def select_ensemble(select_np_index, pred_list, label, ensemble_fn=avg_fn):
    """

    Args:
        select_np_index:
        pred_list:
        label:
        ensemble_fn:

    Returns:

    """
    best_acc = 0
    ensemble_logit = 0
    ensemble_list = []
    num_averaged = 0
    for i in range(len(select_np_index)):
        avg_logit = ensemble_fn(ensemble_logit, pred_list[select_np_index[i]], num_averaged)
        avg_acc = get_metric(avg_logit, label)
        print("i:{}, select_np_index[i]:{} avg_acc:{}, best_acc:{}， num_averaged:{}".format(i, select_np_index[i], avg_acc, best_acc, num_averaged))
        ensemble_list.append(select_np_index[i])
        best_acc = avg_acc
        ensemble_logit = avg_logit
        num_averaged += 1
    print("best acc:{}, ensemble_list:{}".format(best_acc, ensemble_list))

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
    single_random_times = args.single_random_times
    experi_times = args.experi_times
    put_back = int(args.put_back)

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
        print("sort metric_list:{}".format(metric_np[metric_np_index]))
        greedy_ensemble(metric_np_index, pred_list, label)
    elif ensemble_mode == "greedy_brute_force":
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
        print("sort metric_list:{}".format(metric_np[metric_np_index]))
        greedy_brute_force_ensemble(metric_np_index, pred_list, label)
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
        print("sort metric_list:{}".format(metric_np[metric_np_index]))
        random_greedy_ensemble(metric_np_index, pred_list, label, random_times=single_random_times, experi_times=experi_times, put_back=put_back)
    elif ensemble_mode == "random_greedy_v2":
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
        print("sort metric_list:{}".format(metric_np[metric_np_index]))
        random_greedy_v2_ensemble(metric_np_index, pred_list, label, experi_times=experi_times)
    elif ensemble_mode == "swa_all_ensemble":
        # 顺序
        select_index = np.array(list(range(len(pred_pkl_paths))))
        select_ensemble(select_index, pred_list, label, ensemble_fn=swa_avg_fn)
    elif ensemble_mode == "avg_step_all_ensemble":
        # 顺序
        select_index = np.array(list(range(len(pred_pkl_paths))))
        select_ensemble(select_index, pred_list, label, ensemble_fn=avg_fn)

    else:
        raise NotImplementedError




if __name__ == '__main__':
    main()