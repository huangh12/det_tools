#coding:utf-8
'''
Description: This file is for coco-way detection evaluation.
author: he.huang
'''

import init_paths
import logging
logging.getLogger().setLevel(logging.INFO)
import copy
from utils.load_roidb import filter_roidb
from utils.coco_eval import COCOEval
from utils.gen_cocoformat_gt_json import gen_cocoformat_gt_json
from utils.det_eval import evaluate_recall
import cPickle
import os
from config import config as cfg
from config import clsid2clsname, clsid2catid
import numpy as np


def load_roidb(path):
    with open(path, 'rb') as fn:
        roidb = cPickle.load(fn)
    # check id key
    assert 'id' in roidb[0], 'Images are distinguised from id (instead of image name), so id must exist!'
    id_list = [_['id'] for _ in roidb]
    assert len(set(id_list)) == len(id_list), 'id must be unique!'
    return roidb, id_list


if __name__ == "__main__":

    # mkdir to save cache
    if os.path.exists(cfg.cache_dir):
        os.system('rm -rf %s' %cfg.cache_dir)
        logging.info('rm -rf %s' %cfg.cache_dir)
    os.makedirs(cfg.cache_dir)
    logging.info('mkdir in %s to save cache (e.g., GT/result json files)' %cfg.cache_dir)


    # load prediction roidb
    pred_roidb, id_list = load_roidb(cfg.dataset.predict_roidb)
    logging.info('loading %d images from %s for evaluation.' %(len(pred_roidb), cfg.dataset.predict_roidb))
    pred_roidb, choose_inds = filter_roidb(pred_roidb, cfg.filter_strategy, need_inds=True)
    logging.info('sampling %d images for evaluation.' %len(pred_roidb))
    choose_ids = [id_list[_] for _ in choose_inds]

    for task_type in cfg.eval_task_type:
        if not os.path.isfile(cfg.dataset.coco_format_json):
            logging.info('Generate %s coco-format json file using gt roidb...' %task_type)
            gt_roidb, _ = load_roidb(cfg.dataset.gt_roidb)
            cfg.dataset.coco_format_json[task_type] = \
                gen_cocoformat_gt_json(gt_roidb, clsid2clsname, clsid2catid, savedir=cfg.cache_dir)


    def _load_and_check_coco(anno_path, imageset_index):
        task_to_cls = cfg.dataset.task_to_cls if 'task_to_cls' in cfg.dataset else None
        imdb = COCOEval(anno_path, task_to_cls)
        assert set(choose_ids) == (set(choose_ids) & set(imdb.imageset_index)), \
            'the choose_ids should be a subset of imdb.imageset_index'
        imdb.imageset_index = choose_ids
        imdb.num_images = len(imdb.imageset_index)
        if imageset_index is None:
            imageset_index = copy.deepcopy(imdb.imageset_index)
        else:
            for i, j in zip(imageset_index, imdb.imageset_index):
                assert i == j
        return imdb, imageset_index

    imdb = None
    imageset_index = None
    imdb, imageset_index = _load_and_check_coco(cfg.dataset.coco_format_json, imageset_index)
    seg_imdb = None
    if 'seg' in cfg.eval_task_type:
        seg_imdb, imageset_index = _load_and_check_coco(cfg.dataset.coco_format_json_seg, imageset_index)
    assert imageset_index is not None

    def eval_func(**kwargs):
        task_type = cfg.eval_task_type
        if 'rpn' in task_type and cfg.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if cfg.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(gt_roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'det' in task_type:
            imdb.evaluate_detections(kwargs['all_boxes'], res_folder=cfg.cache_dir, alg='det')
        if 'seg' in task_type:
            seg_imdb.evaluate_stuff(kwargs['all_seg_results'], res_folder=cfg.cache_dir, alg='seg')
        if 'kps' in task_type:
            imdb.evaluate_keypoints(kwargs['all_kps_results'], res_folder=cfg.cache_dir, alg='kps')
        if 'mask' in task_type:
            imdb.evalute_mask(kwargs['all_mask_boxes'], kwargs['all_masks'],
                              binary_thresh=cfg.mask_binary_thresh, res_folder=cfg.cache_dir, alg='mask')
        if 'densepose' in task_type:
            imdb.evalute_densepose(kwargs['all_densepose_boxes'], kwargs['all_densepose'], res_folder=cfg.cache_dir, alg='densepose')
        logging.info('Finish evaluation!')
    
    kwargs = dict()
    # prepare all_boxes
    if 'det' in task_type:
        all_boxes = [[[] for _ in range(imdb.num_images)] for _ in range(imdb.num_classes)]
        for i, r in enumerate(pred_roidb):
            for cls, box in zip(r['classes'], r['boxes']):
                assert cls > 0
                if all_boxes[cls][i] == []:
                    all_boxes[cls][i] = box.reshape(-1, 5) 
                else:
                    all_boxes[cls][i] = np.vstack([all_boxes[cls][i], box.reshape(-1, 5) ])
        kwargs['all_boxes'] = all_boxes

    # prepare all_proposals
    # prepare all_seg_results
    
    # prepare all_kps_results
    if 'kps' in task_type:
        all_kps_results = [_['keypoints'].tolist() for _ in pred_roidb]
        kwargs['all_kps_results'] = all_kps_results

    # prepare all_mask_boxes
    # prepare ...

    eval_func(**kwargs)
