import os
import json
import numpy as np
import datetime


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):                                 
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def gen_cocoformat_gt_json(gt_roidb, clsid2clsname, clsid2catid, savedir):
    """Generate coco-format json file from gt_roidb. 
    
    :param gt_roidb  (list)     : input gt_roidb, it must have following key-value items:
                                  'image', 'id', 'width', 'height', 'boxes', 'gt_classes'    
           clsid2clsname  (dict)     : a dict map cls id to cls name
           clsid2catid  (dict)     : a dict map cls id to cat id
           savedir (str)   :         directory to save generated json file
    :return: gt_json_f (str)       : generated file path
    """
    dataset = {u'categories':[], u'images':[], u'annotations':[]}

    categories = dataset[u'categories']
    for clsid, catid in clsid2catid.items():
        categories.append({
            u'supercategory': None,
            u'id': catid,
            u'name': clsid2clsname[clsid]
        })
    images = dataset[u'images']
    annotations = dataset[u'annotations']
    id_ = 0
    for r in gt_roidb:
        images.append({
            u'file_name': r['image'],
            u'id': r['id'],
            u'width': r['width'],
            u'height': r['height']
        })
        for idx, box in enumerate(r['boxes']):
            box = box.astype(np.float32)
            ann = ({
                u'image_id': r['id'],
                u'bbox': [box[0],box[1],box[2]-box[0]+1,box[3]-box[1]+1],
                u'area': r['area'][idx],
                u'category_id': clsid2catid[abs(r['gt_classes'][idx])],
                u'id': id_,
                u'ignore': True if r['gt_classes'][idx] < 0 else False
            })
            # check the consistency
            if 'ignore' in r:
                assert (r['gt_classes'][idx] < 0) == r[u'ignore'][idx], \
                'Warning: The gt_class of ignored box should be negative!'
            ann[u'iscrowd'] = r['iscrowd'][idx] if 'iscrowd' in r else ann[u'ignore']
            ann[u'iscrowd'] = bool(ann[u'iscrowd']) | ann[u'ignore']
            if 'segmentation' in r:
                ann[u'segmentation'] = r['segmentation']
            annotations.append(ann)
            id_ += 1

    import time
    nowTime=str(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())))
    gt_json_f = os.path.join(savedir, '%s.json' %nowTime)
    with open(gt_json_f, 'w') as f:
        json.dump(dataset, f, cls=MyEncoder, sort_keys=True, indent=4)
    print('Generate coco-format json to %s' % gt_json_f)
    return gt_json_f


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Generate coco-format json file from gt_roidb')
    parser.add_argument('--gt_roidb', help='', required=True, type=list)
    parser.add_argument('--clsid2clsname', help='', required=True, type=dict)
    parser.add_argument('--clsid2catid', help='', required=True, type=dict)
    parser.add_argument('--savedir', help='', default='./tmp', type=str)
    parser = parser.parse_args() 
    return parser


if __name__ == "__main__":
    args = parse_args()
    gt_roidb = args.gt_roidb
    clsid2clsname = args.clsid2clsname
    clsid2catid = args.clsid2catid
    savedir = args.savedir

    gen_cocoformat_gt_json(gt_roidb, clsid2clsname, clsid2catid, savedir)
