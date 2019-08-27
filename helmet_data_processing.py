#coding:utf-8
'''
Description: This file is to process the raw annotated json.
             Specifically, 
             1. Split the raw json into raw_train/raw_val json according to the split_ratio.
             2. Generate the train/val roidb from train/val json
             3. Invoking roidb2json function to convert the raw train/val roidb into train/val json for model evaluation

author: he.huang
'''

import os
import random
import numpy as np
import cPickle
import random
import json
import cv2


def empty_dir(dir):
    if os.path.exists(dir):
        os.system('rm -r %s' % dir)
    os.system('mkdir %s' % dir)

category = 'person'
category = 'head'
split_ratio = 0.85

if category == 'person':
    base_dir = '/opt/hdfs/user/he.huang/project/helmet-det/gongdi_data'
else:
    base_dir = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/json_files'
targ_dir = '/mnt/data-1/he.huang/project/helmet-det-x1-job/INT8/pachong_reprocess'
img_dir = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/pachong_images'

all_json = []
image_keys = dict()
ind = 0
with open(os.path.join(base_dir, 'raw_%s.json' %category), 'r') as f:
    line = f.readline()
    while line:
        t = json.loads(line.strip())
        all_json.append(t)
        image_keys[t['image_key']] = ind
        line = f.readline()
        ind += 1

if 'train_img_list.txt' in os.listdir(targ_dir):
    print('Data has been splited, don\'t pollute them!')
    train_imgs = []
    val_imgs = []
    for _ in ['train', 'val']:
        with open(os.path.join(targ_dir, '{}_img_list.txt'.format(_)), 'r') as f:
            line = f.readline()
            while line:
                eval('{}_imgs'.format(_)).append(line.strip())
                line = f.readline()
else:
    imgs = image_keys.keys()
    with open(os.path.join(base_dir, 'img_list.txt'), 'w') as f:
        f.write('\n'.join(imgs))

    num_imgs = len(imgs)
    print('total %d imgs' %num_imgs)

    train_num = int(num_imgs * split_ratio)
    random.shuffle(imgs)
    train_imgs = imgs[:train_num]
    val_imgs = imgs[train_num:]

    for _ in ['train', 'val']:
        with open(os.path.join(targ_dir, '{}_img_list.txt'.format(_)), 'w') as f:
            t = eval('{}_imgs'.format(_))
            f.write('\n'.join(t))
            print('generate %s_img_list.txt' %_)
train_json = []
val_json = []
train_roidb = []
val_roidb = []

for s in ['train', 'val']:
    for _ in eval('{}_imgs'.format(s)):
        eval('{}_json'.format(s)).append(all_json[image_keys[_]])

    str_ = [json.dumps(i) for i in eval('{}_json'.format(s))]
    with open(os.path.join(targ_dir, 'raw_{}_{}.json'.format(category,s)), 'w') as f:
        f.write('\n'.join(str_))

    set_ = eval('{}_json'.format(s))
    num_ = len(set_)
    for i, item in enumerate(set_):
        if i % 100 == 0:
            print('%d/%d' %(i, num_))
            
        img_name = item['image_key']
        assert img_name.endswith('.jpg')

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        roidb = {}
        roidb['image'] = img_name
        roidb['gt_classes'] = []
        # roidb['height'] = int(item['height'])   
        # roidb['width'] = int(item['width'])   
        roidb['height'] = img.shape[0]
        roidb['width'] = img.shape[1]

    
        boxes = []
        if category in item:
            for obj in item[category]: #找到root节点下的所有object节点 
                assert str(obj['attrs']['ignore']) in ['yes', 'no']
                if category == 'person':
                    ignore = obj['attrs']['ignore']=='yes' or \
                         obj['attrs']['occlusion']=='invisible' or\
                         obj['attrs']['hat'] == 'unknown'
                    hat_key_name = 'hat'
                else:
                    ignore = obj['attrs']['ignore']=='yes' or \
                         obj['attrs']['has_hat'] == 'unknown'                    
                    hat_key_name = 'has_hat'
                if ignore:
                    roidb['gt_classes'].append(-1)
                else:
                    if obj['attrs'][hat_key_name] == 'no':
                        roidb['gt_classes'].append(1)
                    elif obj['attrs'][hat_key_name] == 'yes':
                        roidb['gt_classes'].append(2)
                    else:
                        import pdb; pdb.set_trace()
                        assert False, 'unknown hat attr!'
                bndbox = obj['data']      #子节点下属性bndbox的值 
                xmin = np.minimum( roidb['width']-1, np.maximum(0, bndbox[0]) )
                ymin = np.minimum( roidb['height']-1, np.maximum(0, bndbox[1]) )
                xmax = np.maximum(0, np.minimum( roidb['width']-1, bndbox[2]) )
                ymax = np.maximum(0, np.minimum(roidb['height']-1, bndbox[3]) )

                assert xmin<=roidb['width']-1
                assert ymin<=roidb['height']-1
                assert xmax>=0
                assert ymax>=0
                boxes.append([xmin, ymin, xmax, ymax])

        roidb['boxes'] = np.array(boxes, dtype=np.float32).reshape((-1,4))
        roidb['gt_classes'] = np.array(roidb['gt_classes'], dtype=np.int32)

        eval(s+'_roidb').append(roidb)

roidbs_dir = os.path.join(targ_dir, 'roidbs_head')
empty_dir(roidbs_dir)

for s in ['train', 'val']:
    save_path = os.path.join(roidbs_dir, s+'.pkl')
    with open(save_path, 'wb') as f:
        cPickle.dump(eval(s+'_roidb'), f, cPickle.HIGHEST_PROTOCOL)
        print('save to {}'.format(save_path))
        output_json_path = os.path.join(targ_dir, s+'.json')
    os.system('python roidb2json.py --is_gt_roidb True' + 
    ' --input_roidb_path %s --output_json_path %s' %(save_path, output_json_path))
    print('convert %s roidb to %s json' %(s, s))






