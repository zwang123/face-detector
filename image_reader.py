import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import random
import tensorflow as tf
import multiprocessing as mp
from functools import partial
from itertools import product

from scipy.ndimage import zoom

# 50, 4, 200 are connected

def sample_images(img, lbl, ratio=5):

    """ Divide the images into a training set and a test set """

    n = len(img)
    tmask = np.array(random.sample(range(n), n // (ratio+1)), dtype=int)
    #np.save("tmask", tmask)
    timg = img[tmask].copy()
    tlbl = lbl[tmask].copy()
    img = np.delete(img, tmask, axis=0)
    lbl = np.delete(lbl, tmask, axis=0)

    return (img, lbl), (timg, tlbl)

def load_zoom(idx, fnames, saved=False,
        piecesize=50,
        savepre={'img' : 'npy/zoomed', 'lbl' : 'npy/zoomlbl'}):

    """ Load data from npy or from images and zoom """

    imgname = '{}{}.npy'.format(savepre['img'], idx)
    lblname = '{}{}.npy'.format(savepre['lbl'], idx)

    if saved:
        img = np.load(imgname)
        lbl = np.load(lblname)
    else:
        img = []
        lbl = []
        for fname in fnames:
            finished = False
            try:
                fig = plt.imread(fname)
                fig = (zoom(fig, (piecesize/fig.shape[0],
                    piecesize/fig.shape[1], 1)))
                if not (fig.shape == (piecesize, piecesize, 3)):
                    print("zoom", fname, fig.shape)
                #plt.imshow(fig)
                #plt.show()
                #exit()
                img.append(fig)
                finished = True
            except:
                print("except", fname)
                pass
            if finished:
                lbl.append(("HumanHead" in fname,))

        img = np.array(img, ndmin=4, dtype=img[0].dtype)
        lbl = np.array(lbl, ndmin=2, dtype=int)

        np.save(imgname, img)
        np.save(lblname, lbl)

    return img, lbl


def extract_figures(idx, fnames, saved=False, 
        piecesize=50,
        ratio=np.arange(0.1, 1.0, 0.2),
        savepre={'img' : 'npy/fig'}):

    """ Load data from npy or from images and cut into pieces """

    imgname = '{}{}.npy'.format(savepre['img'], idx)

    if saved:
        img = np.load(imgname)
    else:
        img = []
        for fname in fnames:
            try:
                fig = plt.imread(fname)
                for xrat, yrat in product(ratio, repeat=2):
                    xpos = int((fig.shape[0] - piecesize) * xrat)
                    ypos = int((fig.shape[1] - piecesize) * yrat)
                    if xpos + piecesize > fig.shape[0] or \
                       ypos + piecesize > fig.shape[1]:
                           continue
                    img.append(fig[xpos:xpos+piecesize, ypos:ypos+piecesize, :])
            except:
                print("except", fname)
                pass

        img = np.array(img, ndmin=4, dtype=img[0].dtype)

        np.save(imgname, img)

    return img

def load_data(idx, fnames, saved=False, 
        pool=tf.keras.layers.MaxPool2D(4),
        savepre={'img' : 'npy/img', 'lbl' : 'npy/lbl', 'img4' : 'npy/small'}):

    """ Load data from npy or from images and filter through a Pool layer """

    imgname = '{}{}.npy'.format(savepre['img'], idx)
    img4name = '{}{}.npy'.format(savepre['img4'], idx)
    lblname = '{}{}.npy'.format(savepre['lbl'], idx)

    if saved:
        img4 = np.load(img4name)
        try:
            lbl = np.load(lblname)
        except:
            for fname in fnames:
                if len(parse_filename(fname)) != 5:
                    print("parse", fname)
            raise
    else:
        img = []
        lbl = []
        for fname in fnames:
            finished = False
            try:
                img.append(plt.imread(fname))
                finished = True
            except:
                print("except", fname)
                pass
            if finished:
                lbl.append(parse_filename(fname))

        assert(len(lbl) == len(img))

        img = np.array(img, ndmin=4, dtype=img[0].dtype)
        lbl = np.array(lbl, ndmin=2, dtype=lbl[0].dtype)
        img4 = pool(img).numpy()

        np.save(imgname, img)
        np.save(img4name, img4)
        np.save(lblname, lbl)

    #plt.figure(figsize=(10,10))
    #for i in range(25):
    #    plt.subplot(5,5,i+1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.grid(False)
    #    plt.imshow(img4[i])
    #    # The CIFAR labels happen to be arrays,
    #    # which is why you need the extra index
    #plt.show()

    #print(img[0, :8, :8, 0])
    #print(img4[0, :2, :2, 0])

    return img4, lbl

def divide_chunks(arr, chunksize):
    """ divide arr into chunks of chunksize """
    for i in range(0, len(arr), chunksize):
        yield arr[i:i + chunksize]

def load_scenery_from_folder(data_folder = './data/scenery/', saved=False,
        batch=8, **kwargs):

    """ load images from a folder and return only image data 
    with labels fixed at 0"""

    fnames = glob(data_folder + '*')

    with mp.Pool() as p:
        img = (p.starmap(partial(extract_figures, saved=saved, **kwargs), 
            enumerate(divide_chunks(fnames, batch))))

    img = np.concatenate(img)

    print("img", img.shape)

    n = len(img)
    
    return (img, np.zeros((n, 1), dtype=int))

def load_data_from_folder(data_folder = './data/UTKFace/', saved=False,
        load_fxn=load_data,
        batch=512, **kwargs):
        #batch=1024, *args, **kwargs):

    """ load images from a folder and return image data as well as labels """

    fnames = glob(data_folder + '*')

    with mp.Pool() as p:
        rtn = (p.starmap(partial(load_fxn, saved=saved, **kwargs), 
            enumerate(divide_chunks(fnames, batch))))

    img, lbl = zip(*rtn)

    img = np.concatenate(img)
    lbl = np.concatenate(lbl)

    print("img", img.shape)
    print("lbl", lbl.shape)

    return img, lbl


def parse_filename(fname):
    fname = fname.split('/')[-1].split('.')[0].split('j')[0]
    return np.append((1,), np.array(fname.split('_'), dtype=int))

if __name__ == '__main__':
    #targetName='./data/UTKFace/100_0_0_20170112213500903.jpg.chip.jpg'
    #imgDet = plt.imread(targetName)
    #print(imgDet.shape)
    #print(parse_filename(targetName))
    #load_data_from_folder('data/ddd/')
    #load_data_from_folder()
    #load_data_from_folder(saved=True)
    #load_data(0, glob('data/ddd/*'), True)
    load_zoom(0, glob('data/Image/*/*'), False)
