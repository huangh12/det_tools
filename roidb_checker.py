#coding:utf-8
'''
Description: This file is to check the roidb.pkl.
             Plot the boxes on the image to review the correctness of GT boxes.
             Obtain the statistics of the boxes distribution, including the scale and classes
author: he.huang
'''

import sys
# sys.path.insert(0, '/opt/hdfs/user/he.huang/mxnet-python-binds/mxnet-zongbo')
sys.path.insert(0, '/home/users/he.huang/mxnet-zongbo')

import mxnet as mx
import cv2
import cPickle
import random
import os
import numpy as np
from matplotlib import pyplot as plt

# get adequate color list for different classes
from random import randint
colors = []
for i in range(100):
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))


# draw the boxes
def draw_boxes(im, gt_classes, boxes):
    for cls_ind, box in zip(gt_classes, boxes):
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=colors[cls_ind], thickness=2)
        cv2.putText(im, text='cls%s' %(cls_ind), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                    fontScale=0.8, color=colors[cls_ind], thickness=2)
    return im


def cv2_read_img(read_img_dir, img_name):
    for _ in read_img_dir:
        img_path = os.path.join(_, img_name)
        if os.path.isfile(img_path):
            with open(os.path.join(img_path), 'rb') as f:
                im = f.read()[-2:]
            # if not (im[:2] == b'\xff\xd8' and im[-2:] == b'\xff\xd9'):
            #     print('Not complete image: %s' %img_path)
            # else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            return img
    return None


def rec_read_img(roi_rec, imgrec, rgb=False):
    _, img = mx.recordio.unpack_img(imgrec[roi_rec['imgrec_id']].read_idx(roi_rec['imgrec_idx']), cv2.IMREAD_COLOR)
    if rgb:
        img = img[:, :, ::-1]
    return img


def merge_roidb(roidb_list):
    roidb = roidb_list[0]
    for r in roidb_list[1:]:
        roidb.extend(r)
    return roidb


def load_roidb(roidb_path_list, imglst_path_list=None):
    roidb_list = []
    if roidb_path_list is not None:
        for roidb_path in roidb_path_list:
            with open(roidb_path, 'rb') as fn:
                roidb = cPickle.load(fn)
            roidb_list.append(roidb)

    if imglst_path_list is not None:
        add_roidb_imgrec_idx(roidb_list, imglst_path_list)

    roidb = merge_roidb(roidb_list)
    return roidb


def add_roidb_imgrec_idx(roidb_list, imglst_path_list):
    assert len(roidb_list) == len(imglst_path_list)
    for i, roidb in enumerate(roidb_list):
        img_list = {}
        #print imglst_path_list[i]
        with open(imglst_path_list[i], 'r') as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                img_name = line[-1].decode('utf-8')
                assert img_name not in img_list
                img_list[img_name] = int(line[0])

        for roi_rec in roidb:
            img_name = roi_rec['image']
            if img_name not in img_list:
                img_name = os.path.basename(roi_rec['image'])
                assert img_name in img_list, '{} {}'.format(img_name, roi_rec['image'])
            roi_rec['imgrec_id'] = i
            roi_rec['imgrec_idx'] = img_list[img_name]



checker_roidb_path = '/home/users/he.huang/project/HDS_TOOLS/552100_6/roidbs_head/train.pkl'
read_img_dir = ['/home/users/he.huang/project/HDS_TOOLS/552100_6/hangjing_921_1fps',]

# checker_roidb_path = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/public_data/roidbs/trainval.pkl'
# checker_roidb_path = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/data_from_yuhao/trainval.pkl'
# checker_roidb_path = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/gongdi_data/roidbs_head/train.pkl'

# write_img_dir = 'none'

# checker_roidb_path = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/roidbs_head/val.pkl'

# checker_roidb_path = ['val2017_head.pkl']
# checker_lst_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/val2017.lst']
# checker_idx_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/val2017.idx']
# checker_rec_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/val2017.rec']

# checker_roidb_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/roidbs/train2017.pkl']
# checker_lst_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/train2017.lst']
# checker_idx_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/train2017.idx']
# checker_rec_path = ['/opt/hdfs/user/he.huang/common/dataset/cocohead/images_lst_rec/train2017.rec']

# checker_roidb_path = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/roidbs_head/train.pkl'
# read_img_dir = ['/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/pachong_images/']
# write_img_dir = '/mnt/data-1/he.huang/project/helmet-det-x1-job/helmet_det_x1/show_pachong_imgs'

# checker_roidb_path = ['/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/wider_train_pred_head.pkl']
# checker_lst_path = ['/opt/hdfs/user/he.huang/project/wider_face_2019/dataset/WiderFace2019/images_lst_rec/train.lst']
# checker_idx_path = ['/opt/hdfs/user/he.huang/project/wider_face_2019/dataset/WiderFace2019/images_lst_rec/train.idx']
# checker_rec_path = ['/opt/hdfs/user/he.huang/project/wider_face_2019/dataset/WiderFace2019/images_lst_rec/train.rec']
write_img_dir = './show'

rec_read = False
cv2_read = not rec_read
vis = True
vis_num = 100

if vis:
    if os.path.exists(write_img_dir):
        os.system('rm -rf %s' %write_img_dir)
    os.makedirs(write_img_dir)

if rec_read:
    roidb = load_roidb(checker_roidb_path, checker_lst_path)
    if vis:
        imgrec = []
        for imgidx_path, imgrec_path in zip(checker_idx_path, checker_rec_path):
            imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))
else:
    with open(checker_roidb_path,'r') as f:
        roidb = cPickle.load(f)
print('load roidb with %d images' %len(roidb))


random.shuffle(roidb)
cnt = 0
non_empty_cnt = 0
invalid_cnt = 0
invalid_img = []
invalid_cls_ind = []
boxClsStat = {}
boxRatioStat = []
boxScaleStat = []
pick_roidb = []
random.shuffle(roidb)
for i, r in enumerate(roidb):

    print(i)
    boxes_ratio = (r['boxes'][:,3]-r['boxes'][:,1]) / (r['boxes'][:,2]-r['boxes'][:,0])
    boxes_scale = np.sqrt((r['boxes'][:,3]-r['boxes'][:,1] + 1) * (r['boxes'][:,2]-r['boxes'][:,0]+1))
    boxRatioStat.extend( (boxes_ratio).tolist() )
    boxScaleStat.extend( (boxes_scale).tolist() )

    # for _ in r['gt_classes']:
    #     if _ not in [-1,1,2]:
    #         invalid_cnt+=1
    #         invalid_img.append(img_path)
    #         invalid_cls_ind.append(_)
    #         print("Unexpected class id.")
    #         import pdb; pdb.set_trace()

    if 1 in r['gt_classes'] or 2 in r['gt_classes']:
        non_empty_cnt += 1

    unique, counts = np.unique(r['gt_classes'], return_counts=True)
    for u, c in zip(unique, counts):
        if u not in boxClsStat:
            boxClsStat[u] = c
        else:
            boxClsStat[u] += c

    if vis:
        img = None
        if cv2_read:
            img_name = os.path.basename(r['image'])
            img = cv2_read_img(read_img_dir, img_name)
        if rec_read:
            img_name = os.path.basename(r['image'])
            img = rec_read_img(r, imgrec)
        if img is None:
            raise ValueError("Read image fail!")
        
        if r['height'] != img.shape[0] or \
           r['width'] != img.shape[1]:
           print(r['image'], (r['height'], r['width']))
           print(img.shape)
           cv2.imwrite(r['image'], img)
           print('save as %s' %r['image'])
           import pdb; pdb.set_trace()

        img = draw_boxes(img.copy(), r['gt_classes'], r['boxes'])
        save_path = os.path.join(write_img_dir, os.path.basename(r['image']))
        cv2.imwrite(save_path, img)
        print('save plotted im to %s' %(save_path))
        if i == vis_num-1:
            break

print(boxClsStat)
print(non_empty_cnt)

# s1 = [_ for _ in boxRatioStat if _>0 and _<100]
# s2 = [_ for _ in boxScaleStat if _>0 and _<999999]

# plt.hist(s1, bins=100, density=True, range=[0,10])
# plt.savefig('ratio.png')

# import cPickle
# with open('boxRatioStat.pkl', 'w') as f:
#     cPickle.dump(boxRatioStat, f, cPickle.HIGHEST_PROTOCOL)

# with open('boxScaleStat.pkl', 'w') as f:
#     cPickle.dump(boxScaleStat, f, cPickle.HIGHEST_PROTOCOL)

print('DONE!')


