#coding:utf-8
'''
Description: This file is to process the raw annotated json.
             Specifically, 
             1. Split the raw json into raw_train/raw_val json.
             2. Generate the train/val roidb from train/val json

author: he.huang
'''

import os
import random
import numpy as np
import cPickle
import random
import json
import cv2
import shutil

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

raw_json_path = '/home/users/he.huang/project/IR_data/face_hand.json'
save_dir = '/home/users/he.huang/project/IR_data/tmp'
make_dir(save_dir)
shutil.copy(raw_json_path, save_dir)
cat2clsid = {'face': 1}

val_num = 3000

all_json = []
image_keys = dict()
ind = 0
with open(os.path.join(raw_json_path), 'r') as f:
    line = f.readline()
    while line:
        t = json.loads(line.strip())
        all_json.append(t)
        image_keys[t['image_key']] = ind
        line = f.readline()
        ind += 1

if 'train_img_list.txt' in os.listdir(save_dir):
    print('Warning: Data has been splited, don\'t pollute them!')
    train_imgs = []
    val_imgs = []
    for _ in ['train', 'val']:
        with open(os.path.join(save_dir, '{}_img_list.txt'.format(_)), 'r') as f:
            line = f.readline()
            while line:
                eval('{}_imgs'.format(_)).append(line.strip())
                line = f.readline()
else:
    imgs = image_keys.keys()
    with open(os.path.join(save_dir, 'img_list.txt'), 'w') as f:
        f.write('\n'.join(imgs))

    num_imgs = len(imgs)
    print('total %d imgs' %num_imgs)

    random.shuffle(imgs)
    train_imgs = imgs[:-val_num]
    val_imgs = imgs[-val_num:]

    for _ in ['train', 'val']:
        with open(os.path.join(save_dir, '{}_img_list.txt'.format(_)), 'w') as f:
            t = eval('{}_imgs'.format(_))
            f.write('\n'.join(t))
            print('generate %s_img_list.txt' %_)

train_json, val_json = [], []
train_roidb, val_roidb = [], []
old2new_name = {}

cnt = 0
for s in ['train', 'val']:
    for _ in eval('{}_imgs'.format(s)):
        eval('{}_json'.format(s)).append(all_json[image_keys[_]])

    str_ = [json.dumps(i) for i in eval('{}_json'.format(s))]
    with open(os.path.join(save_dir, 'raw_{}.json'.format(s)), 'w') as f:
        f.write('\n'.join(str_))

    set_ = eval('{}_json'.format(s))
    num_ = len(set_)
    for i, item in enumerate(set_):
        if i % 100 == 0:
            print('%d/%d' %(i, num_))
            
        oldname = item['image_key']
        newname = '%09d' %cnt + os.path.splitext(oldname)[1]
        old2new_name[oldname] = newname

        roidb = {}
        roidb['image'] = newname
        roidb['id'] = cnt
        roidb['gt_classes'] = []
        roidb['height'] = int(item['height'])   
        roidb['width'] = int(item['width'])   
        cnt += 1

        boxes = []
        for cat, clsid in cat2clsid.items():
            if cat not in item:
                continue
            for obj in item[cat]: 
                assert str(obj['attrs']['ignore']) in ['yes', 'no']
                # ignore = obj['attrs']['ignore']=='yes' or \
                #         obj['attrs']['occlusion']=='invisible'
                ignore = obj['attrs']['ignore']=='yes'
                if ignore:
                    roidb['gt_classes'].append(-1 * clsid)
                else:
                    roidb['gt_classes'].append(1 * clsid)
                bndbox = obj['data']      
                xmin = np.minimum( roidb['width']-1, np.maximum(0, bndbox[0]) )
                ymin = np.minimum( roidb['height']-1, np.maximum(0, bndbox[1]) )
                xmax = np.maximum(0, np.minimum( roidb['width']-1, bndbox[2]) )
                ymax = np.maximum(0, np.minimum(roidb['height']-1, bndbox[3]) )

                assert xmin <= roidb['width']-1
                assert ymin <= roidb['height']-1
                assert xmax >= 0
                assert ymax >= 0
                boxes.append([xmin, ymin, xmax, ymax])

        roidb['boxes'] = np.array(boxes, dtype=np.float32).reshape((-1,4))
        roidb['gt_classes'] = np.array(roidb['gt_classes'], dtype=np.int32)

        eval(s+'_roidb').append(roidb)

roidbs_dir = os.path.join(save_dir, 'roidbs')
make_dir(roidbs_dir)

for s in ['train', 'val']:
    save_path = os.path.join(roidbs_dir, s+'.pkl')
    with open(save_path, 'wb') as f:
        cPickle.dump(eval(s+'_roidb'), f, cPickle.HIGHEST_PROTOCOL)
        print('save to {}'.format(save_path))

# save the name mapping relation
save_path = os.path.join(save_dir, 'old2new_name.pkl')
with open(save_path, 'wb') as f:
    print('save to {}'.format(save_path))
    cPickle.dump(old2new_name, f, cPickle.HIGHEST_PROTOCOL)