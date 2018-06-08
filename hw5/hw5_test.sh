mkdir ./model
mkdir ./mydict
mkdir ./mydict/log
wget -O model/cmap.pkl "https://www.dropbox.com/s/bsfxozvpcegohev/new_cmap.pkl?dl=0"
wget -O model/corpus.txt "https://www.dropbox.com/s/vi1vza4uxmcr8li/new_corpus.txt?dl=0"
wget -O model/emb.pkl "https://www.dropbox.com/s/3kt9gmq33e62gjj/emb.pkl?dl=0"
wget -O model/model.h5 "https://www.dropbox.com/s/apma9907s4ao1jh/ensemble_best.h5?dl=0"

python test.py --test_path=$1 --result_path=$2 --max_length=30 --model_path=model/model.h5
