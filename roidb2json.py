'''
Converting the predicted roidb from pkl format to json format
for jinbin's evaluation
'''

#coding:utf-8

import os
import random
import numpy as np
import cPickle
import json
import argparse

name2clsid = {
    'nohelmet': 1,
    'withhelmet': 2,
    'ignore': -1
}

def parse_args():
    parser = argparse.ArgumentParser(description='Convert results from pkl to json')
    parser.add_argument('--is_gt_roidb', help='whether is gt roidb', default='False', type=str)
    parser.add_argument('--input_roidb_path', help='input_roidb_path', default='result.pkl', type=str)
    parser.add_argument('--output_json_path', help='output_json_path', default='result.json', type=str)
    parser = parser.parse_args() 
    return parser

args = parse_args()
assert args.is_gt_roidb in ['True', 'False']
print(args)

IS_GT_ROIDB = args.is_gt_roidb == 'True'
input_roidb_path = args.input_roidb_path
output_json_path = args.output_json_path

with open(input_roidb_path, 'r') as f:
    roidbs = cPickle.load(f)
    print('reading roidbs')

write_str = []
records = []
for r in roidbs:
    rec = dict()
    rec['image_key'] = r['image']
    rec['height'] = r['height']
    rec['width'] = r['width']
    rec['video_index'] = -1,
    rec['video_name'] = 'NoName'

    boxes = r['boxes']
    if IS_GT_ROIDB:
        assert type(boxes) == np.ndarray
    else:
        assert type(boxes) == list

    # extract ignore box in GT
    if IS_GT_ROIDB:
        ignore_boxes = boxes[np.where(r['gt_classes']==name2clsid['ignore'])[0]]
        ignore_objs = []
        for det_result in ignore_boxes:
            bbox = [round(float(d), 6) for d in det_result[:4]]
            obj = {
                'data': bbox,
                'id': -1,
                'track_id': -1,
                'struct_type': 'rect',
                'attrs': {'ignore': 'yes'}
            }
            # set attrs 
            ignore_objs.append(obj)

    # process other boxes
    for bigtype in name2clsid.keys():
        if bigtype == 'ignore':
            continue     
        objs = []
        if IS_GT_ROIDB:
            objs = objs + ignore_objs
            cls_boxes = boxes[np.where(r['gt_classes']==name2clsid[bigtype])[0]]
        else:
            cls_boxes = boxes[name2clsid[bigtype]]
        for det_result in cls_boxes:
            bbox = [round(float(d), 6) for d in det_result[:4]]
            obj = {
                'data': bbox,
                'id': -1,
                'track_id': -1,
                'struct_type': 'rect',
            }
            # set attrs 
            attrs = {}
            if not IS_GT_ROIDB:
                conf = round(float(det_result[4]), 6)
                attrs['score'] = conf
            obj['attrs'] = attrs
            objs.append(obj)
        if len(objs) > 0:
            rec[bigtype] = objs
    records.append(rec)

write_str = [json.dumps(_) for _ in records]
with open(output_json_path, 'w') as f:
    f.write('\n'.join(write_str))
print('dumps to %s' %output_json_path)


