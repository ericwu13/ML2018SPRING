from skimage.io import show, imshow_collection, imshow, imread, imsave
from skimage import viewer
import numpy as np
from utils.io import read_images
import argparse

def print_func(func):
    def wrapper(*args, **kwargs):
        print('plot {}'.format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='PCA - eigenfaces')
    parser.add_argument('--data_dir', type=str, metavar='<#data>', default='Aberdeen')
    parser.add_argument('--image', type=str, metavar='<#image>')
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
def reconstruct(images, image, size, num):
    means = np.mean(images, axis=0)
    image_center = images - means;
    im_list = []
    U, S, V = np.linalg.svd(image_center.T, full_matrices=False)
    #U = np.load('SVD/U.npy')
    #S = np.load('SVD/S.npy')
    weights = np.dot(image - means, U[:,:num])
    recon_face = face_fix(np.dot(weights, U[:,:num].T)+means, size)
    imsave("./reconstruction.jpg", recon_face)



def main(args):
    print("Read images set from \"{}\"".format(args.data_dir))
    images, size = read_images(args.data_dir, True)
    print("Read image from \"{}\"".format(args.image))
    image = (np.array(imread(args.image))).flatten()
    reconstruct(images, image, size, num=args.reconstruct)

if __name__ == "__main__":
    args = parse_args()
    main(args)
