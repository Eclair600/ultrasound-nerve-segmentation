from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

import matplotlib.pyplot as plt
from skimage.transform import resize # from scipy.misc import imresize
import nibabel
from tqdm import tqdm

data_path = 'raw/'
image_rows = 256
image_cols = 256
"""
def read_train_exam(exam_nb):
    image = nibabel.load('../chaos/train/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    mask = nibabel.load('../chaos/train/%02d-T2SPIR-mask.nii.gz'%(exam_nb))
    return image, mask

def read_test_exam(exam_nb):
    image = nibabel.load('../chaos/test/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    return image

#Resize each slice to 256x256
#preserve_range to get the same range of intensity
img_rows,img_cols = 256,256
def preprocess(image):
    #shape (slide,256,256)
    image_ = np.ndarray((image.shape[2],img_rows,img_cols,1))
    for i in range(image.shape[2]):
        image_[i,:,:,0] = resize(image.get_data()[:,:,i],(img_rows,img_cols),mode='reflect',preserve_range=True,
                               anti_aliasing=True)
    print(image_.shape)
    return image_
def create_train_data(train_ids = [1,2,3,5,8,10,13,19]):
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for idx, train_id in tqdm(enumerate(train_ids)):
        image, mask = read_train_exam(train_id)     
        image = preprocess(image)
        mask = preprocess(mask)        
        if idx > 0:
            train_data = np.concatenate((train_data, image),axis=0)
            train_mask = np.concatenate((train_mask, mask),axis=0)
        else:
            train_data = image
            train_mask = mask
    print("Final shape {},{}".format(train_data.shape,train_mask.shape))    
    np.savez_compressed('imgs_train', imgs=train_data,)
    np.savez_compressed('imgs_mask_train', imgs_mask=train_mask)
"""
def load_train_data():
    #path data_train
    imgs = np.load('./dataset/imgs_train_0.npz')['imgs']
    imgs_mask = np.load('./dataset/imgs_mask_train_0.npz')['imgs_mask']

    return imgs, imgs_mask


def create_test_data():
    for idx, test_id in tqdm(enumerate(test_ids)):
        image = read_test_exam(test_id)     
        image = preprocess(image)
        print("img {}".format(image.shape))
        if idx > 0:
            test_data = np.concatenate((test_data, image),axis=0)
        else:
            test_data = image
    print("Final shape {}".format(test_data.shape))    
    np.savez_compressed('imgs_test', imgs=test_data)

def load_test_data():
    imgs_test = np.load('imgs_test.npz')['imgs']
    return imgs_test

if __name__ == '__main__':
    create_train_data()
    create_test_data()
