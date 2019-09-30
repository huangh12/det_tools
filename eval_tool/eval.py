#coding:utf-8
'''
Description: This file is for coco-way detection evaluation.
author: he.huang
'''


import sys
sys.path.insert(0, '..')
import logging
logging.getLogger().setLevel(logging.INFO)
import copy
from utils.load_roidb import filter_roidb
from utils.coco_eval import COCOEval
from utils.det_eval import evaluate_recall
import cPickle
import os
from config import config



def load_coco_test_roidb_eval(config):

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    # get roidb
    with open(config.dataset.predict_roidb, 'rb') as fn:
        roidb = cPickle.load(fn)
        logging.info('loading num images for test: {}'.format(len(roidb)))

    roidb, choose_inds = filter_roidb(roidb, config.filter_strategy, need_inds=True)
    logging.info('total num images for test after sampling: {}'.format(len(roidb)))

    # gt_seglabellst_path = config.dataset.gt_seglabellst_path if 'seg' in config.eval_task_type else None

    if config.dataset.coco_format_json is None or \
                    len(config.dataset.coco_format_json) == 0:
        logging.info('Generate coco-format json file using gt roidb...')
        config.dataset.coco_format_json = dict()
        for task_type in config.eval_task_type:
            config.dataset.coco_format_json[task_type] = gen_coco_json(config.dataset.gt_roidb, savedir=config.cache_dir)

    def _load_and_check_coco(anno_path, imageset_index):
        task_to_cls = config.dataset.task_to_cls if 'task_to_cls' in config.dataset else None
        import pdb; pdb.set_trace()
        imdb = COCOEval(anno_path, task_to_cls)
        imdb.imageset_index = [imdb.imageset_index[i] for i in choose_inds]
        imdb.num_images = len(imdb.imageset_index)
        if imageset_index is None:
            imageset_index = copy.deepcopy(imdb.imageset_index)
        else:
            for i, j in zip(imageset_index, imdb.imageset_index):
                assert i == j
        return imdb, imageset_index

    imdb = None
    imageset_index = None
    imdb, imageset_index = _load_and_check_coco(config.dataset.coco_format_json['det'], imageset_index)
    seg_imdb = None
    if 'seg' in config.eval_task_type:
        seg_imdb, imageset_index = _load_and_check_coco(config.dataset.coco_format_json['seg'], imageset_index)
    assert imageset_index is not None

    def eval_func(**kwargs):
        task_type = config.eval_task_type
        if 'rpn' in task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rcnn' in task_type or 'retinanet' in task_type:
            imdb.evaluate_detections(kwargs['all_boxes'], alg=kwargs['alg'] + '-det')
        if 'seg' in task_type:
            seg_imdb.evaluate_stuff(kwargs['all_seg_results'], alg=kwargs['alg'] + '-seg')
        if 'kps' in task_type:
            imdb.evaluate_keypoints(kwargs['all_kps_results'], alg=kwargs['alg'] + '-kps')
        if 'mask' in task_type:
            imdb.evalute_mask(kwargs['all_mask_boxes'], kwargs['all_masks'],
                              binary_thresh=config.TEST.mask_binary_thresh, alg=kwargs['alg'] + '-mask')
        if 'densepose' in task_type:
            imdb.evalute_densepose(kwargs['all_densepose_boxes'], kwargs['all_densepose'], alg=kwargs['alg'] + '-densepose')
    return roidb, eval_func


if __name__ == "__main__":
    load_coco_test_roidb_eval(config)