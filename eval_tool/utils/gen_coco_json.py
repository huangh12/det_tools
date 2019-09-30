import cPickle
import os

def gen_coco_json(roidb_path, save_dir='./cache'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(roidb_path, 'rb') as fn:
        roidb = cPickle.load(fn)
    
    
    