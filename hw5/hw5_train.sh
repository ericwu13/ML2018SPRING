mkdir ./ckpt
mkdir ./log
python train.py .. --load --train_path=$1 --semi_path=$2 --max_length=30
