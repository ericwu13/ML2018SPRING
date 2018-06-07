mkdir ./model
wget -O model/cmap.pkl "https://www.dropbox.com/s/8lipdpfh1rn6fj9/cmap_3W_30padd.pkl?dl=0"
wget -O model/corpus.txt "https://www.dropbox.com/s/9x3rkn4ujs5ozba/corpus_3W_30padd.txt?dl=0"
wget -O moel/word2vec.pkl "https://www.dropbox.com/s/sce4kk9n2c70wwg/word2vec_3W_30padd.pkl?dl=0"
wget -O moel/model.h5 "https://www.dropbox.com/s/i2e24ek1agi6v1o/ensemble_5_0.839.h5?dl=0"

python test.py --test_path=$1 --result_path=$2 --max_length=30 --model_path=model/model.h5
