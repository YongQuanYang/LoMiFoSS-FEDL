
export PYTHONPATH=./:${PYTHONPATH}

train_data_root="./dataset/imagenet/"
val_data_root="./dataset/imagenet/"
val_list="./dataset/imagenet/val_list.txt"
#train_list="./dataset/imagenet/train_list.txt"
train_list="./dataset/imagenet/train_0.95.txt"
#val_list="./dataset/imagenet/train_0.05.txt"
delimiter=" "
swa_lr=0.001
model="resnet50"
work_dir="./work_dirs/imagenet_${model}_retrain_train0.95_val"
mkdir -p ${work_dir}

########################### train process #############################

CUDA_VISIBLE_DEVICES=4,5,6,7 python  experiments/imagenet/run_swag_imagenet.py \
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
    --lr_init=0.1 \
    --parallel \
    --epochs=120 \
    --save_freq=120 \
    --eval_freq=1 2>&1 | tee -a ${work_dir}/log.txt


########################### evalate origin model process #############################
pretrain_path=${work_dir}/normal_best-0.pt
val_list="./dataset/imagenet/val_list.txt"
#val_list="./dataset/imagenet/train_0.05.txt"
CUDA_VISIBLE_DEVICES=0,1 python  experiments/imagenet/run_swag_imagenet.py \
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
    --pretrain_path=${pretrain_path}\
    --parallel \
    --epochs=10 \
    --save_freq=10 \
    --eval_freq=1 \
    --evaluate \
    --swa \
    --swa_start=0 \
    --swa_lr=${swa_lr} \
    --swa_freq=4 2>&1 | tee -a ${work_dir}/eval_log.txt



