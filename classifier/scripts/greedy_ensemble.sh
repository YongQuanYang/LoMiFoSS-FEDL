
export PYTHONPATH=./:${PYTHONPATH}
pred_path=./work_dirs/imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index4_v3/res/norm-*.pkl
label_path=./work_dirs/imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index4_v3/res/norm-1.pkl

CUDA_VISIBLE_DEVICES=0,1 python  experiments/imagenet/greedy_weighted_ensemble.py \
    --pred_path=${pred_path} \
    --label_path=${label_path}
#
