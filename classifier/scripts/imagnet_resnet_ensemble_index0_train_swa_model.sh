
export PYTHONPATH=./:${PYTHONPATH}

train_data_root="./dataset/imagenet/"
val_data_root="./dataset/imagenet/"
val_list="./dataset/imagenet/val_list.txt"
#train_list="./dataset/imagenet/train_list.txt"
train_list="./dataset/imagenet/train_0.95.txt"
#val_list="./dataset/imagenet/train_0.05.txt"
#val_list="./dataset/imagenet/val_list_test.txt"
#train_list="./dataset/imagenet/val_list_test.txt"
delimiter=" "
swa_lr=0.001
model="resnet50"
ensemble_index=0
train_swa_model=1
pretrain_path=./work_dirs/imagenet_resnet50_retrain_train0.95_val/normal_best-0.pt
work_dir="./work_dirs/imagenet_retrain_train0.95_val_${model}_swa_finetune_swalr${swa_lr}_constant_lr_2gpu_ensemble_index${ensemble_index}_train_swa_model"
mkdir -p ${work_dir}

#
#
#CUDA_VISIBLE_DEVICES=1,2 python  experiments/imagenet/run_swag_imagenet.py \
#    --dir=${work_dir} \
#    --model=${model} \
#    --num_classes=1000 \
#    --train_data_root=${train_data_root} \
#    --val_data_root=${val_data_root} \
#    --val_list=${val_list} \
#    --train_list=${train_list} \
#    --num_workers=8 \
#    --delimiter=${delimiter} \
#    --batch_size=256 \
#    --pretrained \
#    --pretrain_path=${pretrain_path} \
#    --parallel \
#    --epochs=10 \
#    --train_swa_model=${train_swa_model} \
#    --save_freq=10 \
#    --eval_freq=1 \
#    --swa \
#    --swa_start=0 \
#    --ensemble_index=${ensemble_index} \
#    --swa_lr=${swa_lr} \
#    --swa_freq=4 2>&1 | tee -a ${work_dir}/log.txt



#evaluate
#swa_resume=${work_dir}/swag_best-0.pt
swa_resume=./work_dirs/imagenet_retrain_train0.95_val_resnet50_swa_finetune_swalr0.001_constant_lr_2gpu_ensemble_index0_v2/swag_best-0.pt
#val_list="./dataset/imagenet/val_list.txt"
val_list="./dataset/imagenet/train_0.05.txt"
CUDA_VISIBLE_DEVICES=2,3 python  experiments/imagenet/run_swag_imagenet.py \
    --dir=${work_dir} \
    --model=${model} \
    --num_classes=1000 \
    --train_data_root=${train_data_root} \
    --val_data_root=${val_data_root} \
    --val_list=${val_list} \
    --train_list=${train_list} \
    --num_workers=8 \
    --delimiter=${delimiter} \
    --batch_size=256 \
    --pretrained \
    --swa_resume=${swa_resume} \
    --parallel \
    --epochs=10 \
    --save_freq=10 \
    --eval_freq=1 \
    --evaluate \
    --swa \
    --swa_start=0 \
    --ensemble_index=${ensemble_index} \
    --swa_lr=${swa_lr} \
    --swa_freq=4 2>&1 | tee -a ${work_dir}/eval_log.txt



