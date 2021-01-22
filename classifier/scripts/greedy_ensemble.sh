
export PYTHONPATH=./:${PYTHONPATH}

data="res" #val data
#data="test_res" # test data

pred_path=./work_dirs/imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index4_v4_20210119/${data}/norm-*.pkl
label_path=./work_dirs/imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index4_v4_20210119/${data}/norm-1.pkl

#CUDA_VISIBLE_DEVICES=0,1 python  experiments/imagenet/greedy_weighted_ensemble.py \
#    --pred_path=${pred_path} \
#    --label_path=${label_path} \
#    --ensemble_mode="greedy"

#CUDA_VISIBLE_DEVICES=0,1 python  experiments/imagenet/greedy_weighted_ensemble.py \
#    --pred_path=${pred_path} \
#    --label_path=${label_path} \
#    --ensemble_mode="random_greedy" \
#    --random_times=100

CUDA_VISIBLE_DEVICES=0,1 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path} \
    --ensemble_mode="select" \
    --select_index 14 11 14 18 14 20 17 17 14 11

