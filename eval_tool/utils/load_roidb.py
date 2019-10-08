import os
import cPickle
import logging
import numpy as np
import chardet


def filter_roidb(roidb, filter_strategy, need_inds=False):
    all_choose_inds = range(len(roidb))

    def filter_roidb_func(choose_inds, filter_name, filter_func):
        if filter_name in filter_strategy and filter_strategy[filter_name]:
            num = len(choose_inds)
            choose_inds = [i for i in choose_inds if not filter_func(roidb[i])]
            num_after = len(choose_inds)
            logging.info('filter %d %s roidb entries: %d -> %d' % (num - num_after, filter_name[7:], num, num_after))
        return choose_inds

    def is_points_as_boxes(entry):
        gt_boxes = entry['boxes']
        width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
        height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
        flag = (width > 1).all() and (height > 1).all()
        return not flag
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_point', is_points_as_boxes)

    def is_empty_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes == 0
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_boxes', is_empty_boxes)

    def is_single_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes <= 1
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_single_boxes', is_single_boxes)

    def is_multi_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes > 1
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_multi_boxes', is_multi_boxes)

    def is_bg_seg_label(entry):
        return entry['seg_is_background']
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_bg_seg_label', is_bg_seg_label)

    def is_empty_boxes_and_bg_seg_label(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes == 0 and entry['seg_is_background']
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_boxes_bg_seg_label', is_empty_boxes_and_bg_seg_label)

    def is_empty_densepose(entry):
        return 'densepose' not in entry
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_densepose', is_empty_densepose)

    def is_empty_kps(entry):
        scores = entry['keypoints'][:, 2::3]
        scores = np.sum(scores, axis=1)
        keep = np.where(scores != 0)[0]
        entry['keypoints'] = entry['keypoints'][keep, :]
        entry['boxes'] = entry['boxes'][keep, :]
        if 'gt_classes' in entry:
            entry['gt_classes'] = entry['gt_classes'][keep]
        return np.sum(scores) == 0
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_kps', is_empty_kps)

    def is_any_unvis_kps(entry):
        scores = entry['keypoints'][:, 2::3]
        return (scores != 2).any()
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_any_unvis_kps', is_any_unvis_kps)

    if 'max_num_images' in filter_strategy:
        max_num_images = filter_strategy['max_num_images']
        if 0 < max_num_images < len(all_choose_inds):
            num = len(all_choose_inds)
            all_choose_inds = all_choose_inds[:max_num_images]
            num_after = len(all_choose_inds)
            logging.info('filter %d roidb entries after max_num_images: %d -> %d' % (num - num_after, num, num_after))

    if 'parts' in filter_strategy:
        part_index = filter_strategy['parts'][0]
        num_parts = filter_strategy['parts'][1]
        assert part_index < num_parts
        num_inds_per_part = (len(all_choose_inds) + num_parts - 1) / num_parts
        num = len(all_choose_inds)
        all_choose_inds = all_choose_inds[part_index*num_inds_per_part: (part_index+1)*num_inds_per_part]
        num_after = len(all_choose_inds)
        logging.info('filter %d roidb entries after parts: %d -> %d' % (num - num_after, num, num_after))

    if 'indexes' in filter_strategy:
        start_index = filter_strategy['indexes'][0]
        end_index = filter_strategy['indexes'][1]
        num = len(all_choose_inds)
        assert 0 <= start_index < end_index <= num
        all_choose_inds = all_choose_inds[start_index:end_index]
        num_after = len(all_choose_inds)
        logging.info('filter %d roidb entries after indexes: %d -> %d' % (num - num_after, num, num_after))

    roidb = [roidb[i] for i in all_choose_inds]

    if need_inds:
        return roidb, all_choose_inds
    else:
        return roidb

