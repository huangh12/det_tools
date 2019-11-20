#coding:utf-8
'''
Description: This file is to check the completeness of images, in case of corruption
author: he.huang
'''

import cv2
import cPickle
import os
import glob
import shutil

image_dir = 'xxx/'
DELETE = False
save_corrupt_dir = './corrupt'

if os.path.exists(save_corrupt_dir):
    os.system('rm -rf %s' %save_corrupt_dir)
os.makedirs(save_corrupt_dir)

images = glob.glob(image_dir + '/*.jpg')
num_image = len(images)

for i, img_path in enumerate(images):
    if i % 100 == 0:
        print('%d/%d' %(i,num_image))
    if os.path.isfile(img_path):
        with open(img_path, 'rb') as f:
            _ = f.read()
            check_begin_chars = _[:2]
            check_end_chars = _[-2:]
        if not (check_begin_chars == b'\xff\xd8' and check_end_chars == b'\xff\xd9'):
            print('Not complete image: %s' %img_path)
            targ = os.path.join(save_corrupt_dir, os.path.basename(img_path))
            shutil.copy(img_path, targ)