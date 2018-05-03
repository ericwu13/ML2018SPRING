import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='TSNE clustering with Kmeans')
    parser.add_argument('--PCA', type=int, metavar='<#pca_images>')
    parser.add_argument('--encoder', type=str, metavar='<#encoder_images>')
    parser.add_argument('--tsne', action='store_true')
    return parser.parse_args()
def clustering(encode_images, tsne=False):
        f, (ax1, ax2) = plt.subplots(1, 2)
        if(tsne):
            X_embedded = TSNE(n_components=2, random_state=120).fit_transform(encode_images)
            np.save('tsne.npy', X_embedded)
        else:
            X_embedded = (encode_images)
            X_embedded = np.load('tsne.npy')
        # plot by known label distribution
        ax1.scatter(X_embedded[:5000, 0], X_embedded[:5000, 1], c='b', label='dataset A', s=0.2)
        ax1.scatter(X_embedded[5000:, 0], X_embedded[5000:, 1], c='r', label='dataset B', s=0.2)
        ax1.legend()
        ax1.set_title('known label')

        kmeans = KMeans(n_clusters=2, random_state=1120).fit(X_embedded)
        label = kmeans.labels_
        knn_graph = kneighbors_graph(X_embedded, 20, mode='connectivity', include_self=False)
        model = AgglomerativeClustering(linkage='ward',
                                            connectivity=knn_graph,
                                            n_clusters=2)
        model.fit(X_embedded)
        label = model.labels_
        label_0 = []
        label_1 = []
        for idx, e in enumerate(label):
            if(e):
                label_1.append(X_embedded[idx])
            else:
                label_0.append(X_embedded[idx])
        print("Distribution is ({}, {})".format(len(label_0), len(label_1)))
        #pair = np.concatenate((np.array([label]).T, X_embedded), axis=1)
        #X_embedded = (np.sort(pair, axis=0))[:,1:]
        label_0 = np.array(label_0)
        label_1 = np.array(label_1)

        ax2.scatter(label_1[:, 0], label_1[:, 1], c='b', label='dataset B', s=0.2)
        ax2.scatter(label_0[:, 0], label_0[:, 1], c='r', label='dataset A', s=0.2)
        ax2.legend()
        ax2.set_title('predict label')
        f.savefig('cluster.png')
        f.show();
def main(args):
    if args.PCA is not None:
        images = np.load('data/image.npy')
        visual = np.load('data/visualization.npy')
        print("Clustring with \"PCA_{}\" using Kmeans".format(args.PCA))
        pca = PCA(args.PCA, whiten=True, random_state=120)
        pca.fit(images)
        images = pca.transform(visual)
        clustering(images, args.tsne)
    if args.encoder is not None:
        images = np.load(args.encoder)
        filename = args.encoder.split('/')[-1]
        print("Clustring with \"{}\" using Kmeans".format(filename))
        clustering(images, args.tsne)
if __name__ == "__main__":
    args = parse_args()
    main(args)
