
export PYTHONPATH=./:${PYTHONPATH}

data="res" #val data
#data="test_res" # test data

workspace=imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index4_v4_20210124
#workspace=imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index0_v4_20210124
pred_path=./work_dirs/${workspace}/${data}/norm-*.pkl
label_path=./work_dirs/${workspace}/${data}/norm-1.pkl

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="greedy"

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="random_greedy" \
    --single_random_times=100 \
    --experi_times=20 \
    --put_back=0

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="greedy_brute_force" \
    --single_random_times=100 \
    --experi_times=20 \


CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="random_greedy_v2" \
    --experi_times=21 \

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="swa_all_ensemble" \

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="avg_step_all_ensemble" \

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="select" \
    --select_index 5 1 7

