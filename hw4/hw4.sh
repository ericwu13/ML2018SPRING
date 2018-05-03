python rdu_dim.py --read=auto_encoder.h5 --data=$1
python clustering.py --encoder=auto_encoder.h5.npy --test=$2 --save=$3
