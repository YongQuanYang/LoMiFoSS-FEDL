
export PYTHONPATH=./:${PYTHONPATH}

train_data_root="./dataset/imagenet/"
val_data_root="./dataset/imagenet/"
val_list="./dataset/imagenet/val_list.txt"
train_list="./dataset/imagenet/train_list.txt"
delimiter=" "
swa_lr=0.001
model="resnet50"
work_dir="./work_dirs/imagenet_${model}_swa_finetune_swalr${swa_lr}_constant_lr_2gpu"
mkdir -p ${work_dir}

#
#CUDA_VISIBLE_DEVICES=6,7 python  experiments/imagenet/run_swag_imagenet.py \
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
#    --parallel \
#    --epochs=10 \
#    --save_freq=10 \
#    --eval_freq=1 \
#    --swa \
#    --swa_start=0 \
#    --swa_lr=${swa_lr} \
#    --swa_freq=4 2>&1 | tee -a ${work_dir}/log.txt

