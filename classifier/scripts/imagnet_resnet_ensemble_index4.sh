
export PYTHONPATH=./:${PYTHONPATH}

train_data_root="./dataset/imagenet/"
val_data_root="./dataset/imagenet/"
#train_list="./dataset/imagenet/train_list.txt"
train_list="./dataset/imagenet/train_0.95.txt"
val_list="./dataset/imagenet/val_list.txt"
delimiter=" "
swa_lr=0.001
model="resnet50"
ensemble_index=4
pretrain_path=./work_dirs/imagenet_resnet50_retrain_train0.95_val/normal_best-0.pt
work_dir="./work_dirs/imagenet_retrain_train0.95_val_${model}_swa_finetune_swalr${swa_lr}_constant_lr_2gpu_ensemble_index${ensemble_index}_v4_20210124"
mkdir -p ${work_dir}

########################### train process #############################

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
    --pretrain_path=${pretrain_path} \
    --parallel \
    --epochs=20 \
    --save_freq=1 \
    --eval_freq=1 \
    --swa \
    --swa_start=0 \
    --ensemble_index=${ensemble_index} \
    --swa_lr=${swa_lr} \
    --swa_freq=4 2>&1 | tee -a ${work_dir}/log.txt


########################### evalate swa model process #############################
swa_resume=${work_dir}/swag_best-0.pt
#val_list="./dataset/imagenet/val_list.txt"
val_list="./dataset/imagenet/train_0.05.txt"
CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/run_swag_imagenet.py \
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
    --swa_freq=4 #2>&1 | tee -a ${work_dir}/eval_log.txt


########################### evaluate norm checkpoint and extract result  #############################

#evaluate
mkdir -p ${work_dir}/res
CKPT_INDEX_SET="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
# for val dataset evaluate
val_list="./dataset/imagenet/val_list.txt"
data="res" #val data

# for test dataset evaluate
#val_list="./dataset/imagenet/train_0.05.txt"
#data="test_res" # test data
mkdir -p ${work_dir}/${data}
for CKPT_INDEX in ${CKPT_INDEX_SET}
do
    if [[ ${CKPT_INDEX} == "0" ]];then
        # base model
        pretrain_path=./work_dirs/imagenet_resnet50_retrain_train0.95_val/normal_best-0.pt
        res_path=${work_dir}/${data}/norm-0.pkl
    else
        # checkpoint to eval
        pretrain_path=${work_dir}/norm-${CKPT_INDEX}.pt
        res_path=${work_dir}/${data}/norm-${CKPT_INDEX}.pkl
    fi

CUDA_VISIBLE_DEVICES=4,5 python  experiments/imagenet/run_swag_imagenet.py \
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
    --pretrain_path=${pretrain_path} \
    --parallel \
    --epochs=20 \
    --save_freq=1 \
    --eval_freq=1 \
    --evaluate \
    --ensemble_index=${ensemble_index} \
    --res_path=${res_path} \
    --swa_freq=4 2>&1 | tee -a ${work_dir}/eval_log.txt
done

