import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser(description='TSNE clustering with Kmeans')
    parser.add_argument('--PCA', type=str, metavar='<#pca_images>')
    parser.add_argument('--encoder', type=str, metavar='<#encoder_images>')
    parser.add_argument('--test', type=str, metavar='<#test_path>')
    parser.add_argument('--save', type=str, metavar='<#save_path>')

    return parser.parse_args()


def clustering(tsne_images, filename, save, path='data/test_case.csv'):
        test_data = (pd.read_csv(path, header=0)).values

        kmeans = KMeans(n_clusters=2, random_state=1120).fit(tsne_images)
        label = kmeans.labels_
        count_1 = 0
        count_0 = 0
        for idx, element in enumerate(label):
            if(element == 1):
                count_1 +=1
            else:
                count_0 +=1
        print("label 1: {}, label 0: {}".format(count_1, count_0))
        ID = []
        similarity = []
        for i in range(len(test_data)):
            idx_1 = test_data[i][1]
            idx_2 = test_data[i][2]
            ID.append(i)
            if(label[idx_1] == label[idx_2]):
                similarity.append(1)
            else:
                similarity.append(0)
        columns = ['ID', 'Ans']
        d = np.array([ID,similarity])
        df = pd.DataFrame(data=d.T, columns=['ID', 'Ans'])
        df.to_csv(save, encoding='utf-8', index=False)

    
def main(args):
    path = ""
    save = ""
    if(args.test is not None):
        path = args.test
    if(args.save is not None):
        save = args.save
    if args.PCA is not None:
        images = np.load(args.PCA)
        filename = args.PCA.split('/')[-1]
        print("Clustring with \"{}\" using Kmeans".format(filename))
        np.save('tsne_images/tsne_{}'.format(filename), images)
        clustering(images, filename)
    if args.encoder is not None:
        images = np.load(args.encoder)
        filename = args.encoder.split('/')[-1]
        print("Clustring with \"{}\" using Kmeans".format(filename))
        clustering(images, filename, save, path)
if __name__ == "__main__":
    args = parse_args()
    main(args)
