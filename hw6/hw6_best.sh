mkdir model
wget -O model/user_enc.pkl "https://www.dropbox.com/s/qil1qz7ng85fgfe/user_enc.pkl?dl=0"
wget -O model/movie_enc.pkl "https://www.dropbox.com/s/1nhloawg8znze15/movie_enc.pkl?dl=0"
wget -O model/model.h5 "https://www.dropbox.com/s/oc1i826dmyc7xkn/0.87loss.h5?dl=0"
python test.py --test_path=$1 --result_path=$2 --model_path=model/model.h5
