from os import listdir
from os.path import isfile, join, dirname, realpath
from skimage.io import imread_collection
import numpy as np

base_dir = dirname((dirname(realpath(__file__))))
data_dir = join(base_dir,'data')

def read_images(path='Aberdeen', ispath=False):
    # load all the .jpg images from give file, default is data/Aberdeen/*.jpg
    if(ispath):
        image_dir = join(base_dir, path)
    else:
        image_dir = join(data_dir, path)
    images = np.array(imread_collection(image_dir+"/*.jpg"))# / 255.
    size = np.shape(images[0])
    X = []
    for idx, image in enumerate(images):
        X.append(image.flatten())
    return np.array(X), size

