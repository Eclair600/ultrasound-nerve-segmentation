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

def read_train_exam(exam_nb):
    image = nibabel.load('./chaos/train/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    mask = nibabel.load('./chaos/train/%02d-T2SPIR-mask.nii.gz'%(exam_nb))
    return image, mask

def read_test_exam(exam_nb):
    image = nibabel.load('./chaos/test/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    return image

#Resize each slice to 256x256
#preserve_range to get the same range of intensity
def preprocess(image):
    image_ = np.ndarray((img_rows,img_cols,image.shape[2]))
    for i in range(image.shape[2]):
        image_[:,:,i] = resize(image.get_data()[:,:,i],(img_rows,img_cols),
                               mode='reflect',
                               preserve_range=True,
                               anti_aliasing=True)
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
            train_data = np.concatenate((train_data, image),axis=-1)
            train_mask = np.concatenate((train_mask, mask),axis=-1)
        else:
            train_data = image
            train_mask = mask   
    np.savez_compressed('imgs_train', imgs=train_data,)
    np.savez_compressed('imgs_mask_train', imgs_mask=train_mask)
    print("Training images saved {},{}".format(train_data.shape,train_mask.shape)) 


def load_train_data():
    imgs = np.load('imgs_train.npz')['imgs']
i   mgs_mask = np.load('imgs_mask_train.npz')['imgs_mask']

    return imgs, imgs_mask


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
