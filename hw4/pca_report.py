from skimage.io import show, imshow_collection, imshow
from skimage import viewer
import numpy as np
from utils.io import read_images
import argparse
#import matplotlib.pyplot as plt

def print_func(func):
    def wrapper(*args, **kwargs):
        print('plot {}'.format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='PCA - eigenfaces')
    parser.add_argument('--data_dir', type=str, metavar='<#data>', default='Aberdeen')
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--findk', action='store_true')
    parser.add_argument('--eigenface', type=int, metavar='<#eigenface>')
    parser.add_argument('--reconstruct', type=int, metavar='<#reconstruct>')
    return parser.parse_args()

def face_fix(x, size):
    M = x.reshape(size)
    M -= np.min(M)
    M /= np.max(M)
    eigenface = (M * 255).astype(np.uint8)
    return eigenface


@print_func
def eigenface(images, size, num):
    means = np.mean(images, axis=0)
    image_center = images - means;
    #U, S, V = np.linalg.svd(image_center.T, full_matrices=False)
    #np.save('U.npy', U)
    #np.save('S.npy', S)
    U = np.load('SVD/U.npy')
    S = np.load('SVD/S.npy')
    data = list(zip(U.T, S))
    data =sorted(data, key=lambda tup:(-tup[1], tup[0]))
    U, S = (zip(*data))
    U, S = (np.array(U)).T, np.array(S)
    ready2show = []
    sub_width, sub_height = np.ceil(num / 1).astype(int), 1
    fig = plt.figure(figsize=(9, 9))
    print("Sum of the eigenvalues = \"{}\"".format(np.sum(S)))
    for i in range(num):
        print("No.{} eigenvalue = \"{}\", with ratio \"{}\"".format(i+1, S[i], S[i]/float(np.sum(S))))
        M = U[:,i].reshape(size)
        M -= np.min(M)
        M /= np.max(M)
        eigenface = (M * 255).astype(np.uint8)
        #ready2show.append(eigenface)
        ax = fig.add_subplot(sub_height, sub_width, i+1)
        ax.imshow(eigenface)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    #imshow_collection(ready2show)
    fig.savefig('eigenface.png')
    fig.show()
    #imshow(ready2show[9])
    #show()

@print_func
def reconstruct(images, size, num):
    means = np.mean(images, axis=0)
    image_center = images - means;
    im_list = []
    U = np.load('SVD/U.npy')
    S = np.load('SVD/S.npy')
    #U, S, V = np.linalg.svd(image_center.T, full_matrices=False)
    # choose 0, 10, 43, 197 
    sub_width, sub_height = 4, 2
    fig = plt.figure(figsize=(8, 4))
    weights = np.dot(image_center, U[:,:num])
    '''
    recon_face_1 = face_fix(np.dot(weights[0], U[:,:num].T)+means, size)
    im_list.append((images[0].reshape(size), recon_face_1))
    recon_face_10 = face_fix(np.dot(weights[10], U[:,:num].T)+means, size)
    im_list.append((images[10].reshape(size), recon_face_10))
    recon_face_43 = face_fix(np.dot(weights[43], U[:,:num].T)+means, size)
    im_list.append((images[43].reshape(size), recon_face_43))
    recon_face_197 = face_fix(np.dot(weights[197], U[:,:num].T)+means, size)
    im_list.append((images[197].reshape(size), recon_face_197))
    for i, element in enumerate(im_list):
        ax = fig.add_subplot(sub_height, sub_width, 2*i+1)
        ax.imshow(element[0])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = fig.add_subplot(sub_height, sub_width, 2*i+2)
        ax.imshow(element[1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    fig.savefig('recon.png')
    fig.show()
    '''



def main(args):
    images, size = read_images(args.data_dir)
    if args.eigenface is not None:
        eigenface(images, size, num=args.eigenface)
    if args.reconstruct is not None:
        reconstruct(images, size, num=args.reconstruct)

if __name__ == "__main__":
    args = parse_args()
    main(args)
