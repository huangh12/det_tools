# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import numpy as np
import numpy.random as npr
from chips.chip_generator import chip_generator
import math
import copy_reg
import types
from multiprocessing import Pool
from common.bbox.bbox_transform import *


# Pickle dumping recipe for using classes with multi-processing map
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class chip_worker(object):
    def __init__(self, cfg, chip_size):
        self.valid_ranges = cfg.valid_ranges
        self.scales = cfg.pyramid_scales
        self.chip_size = chip_size
        self.use_cpp = cfg.cpp_chips
        self.chip_stride = cfg.chip_stride
        self.chip_generator = chip_generator(chip_stride=self.chip_stride, use_cpp=self.use_cpp)
        self.use_neg_chips = cfg.use_neg_chips
        self.chip_gt_overlap = cfg.chip_gt_overlap

    def reset(self):
        self.chip_generator = chip_generator(chip_stride=self.chip_stride, use_cpp=self.use_cpp)

    def chip_extractor(self, r):
        width = r['width']
        height = r['height']
        im_size_max = max(width, height)

        # gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]
        gt_boxes = r['boxes'][np.where(r['gt_classes'] > 0)[0], :]

        ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
        hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
        area = np.sqrt(ws * hs)
        ms = np.maximum(ws, hs)

        chip_ar = []
        for i, im_scale in enumerate(self.scales):
            if i == len(self.scales)-1:
                # The coarsest (or possibly the only scale)
                ids = np.where((area >= self.valid_ranges[i][0]))[0]
            elif i == 0:
                # The finest scale (but not the only scale)
                ids = np.where((area < self.valid_ranges[i][1]) & (ws >= 2) & (hs >= 2))[0]
            else:
                # An intermediate scale
                ids = np.where((area >= self.valid_ranges[i][0]) & (area < self.valid_ranges[i][1]))[0]

            cur_chips = self.chip_generator.generate(gt_boxes[ids, :] * im_scale, int(r['width'] * im_scale), int(r['height'] * im_scale),
                                 self.chip_size)
            cur_chips = np.array(cur_chips) / im_scale

            for chip in cur_chips:
                chip_ar.append([chip, im_scale, int(r['height'] * im_scale), int(r['width'] * im_scale)])

        return chip_ar

    def box_assigner(self, r):
        props_in_chips = [[] for _ in range(len(r['crops']))]
        widths = (r['boxes'][:, 2] - r['boxes'][:, 0]).astype(np.int32)
        heights = (r['boxes'][:, 3] - r['boxes'][:, 1]).astype(np.int32)
        max_sizes = np.maximum(widths, heights)
        width = r['width']
        height = r['height']
        
        # if (max_sizes < 0).any():
        #     print '------------\n', r['boxes'][max_sizes<0]            

        # if np.min(r['boxes']) < 0:
        #     print '------------\n', r['boxes'][np.where(r['boxes']<0)[0]] 

        area = np.sqrt(np.maximum(0,widths * heights))
        im_size_max = max(width, height)

        # ** Distribute chips based on the scales
        all_chips = [[] for _ in self.scales]
        all_chip_ids = [[] for _ in self.scales]
        for ci, crop in enumerate(r['crops']):
            for scale_i, s in enumerate(self.scales):
                if (scale_i == len(self.scales) - 1) or s == crop[1]:
                    all_chips[scale_i].append(crop[0])
                    all_chip_ids[scale_i].append(ci)
                    break

        # All chips in each of the scales:
        all_chips = [np.array(chips) for chips in all_chips]
        # The ids of chips in each of the scales:
        all_chip_ids = [np.array(chip_ids) for chip_ids in all_chip_ids]

        # ** Find valid boxes in each scale
        valid_ids = []
        for scale_i, im_scale in enumerate(self.scales):
            if scale_i == len(self.scales) - 1:
                # The coarsest scale (or the only scale)
                ids = np.where((area >= self.valid_ranges[scale_i][0]))[0]
            else:
                ids = np.where((area < self.valid_ranges[scale_i][1]) & (area >= self.valid_ranges[scale_i][0]) &
                               (widths >= 2) & (heights >= 2))[0]
            valid_ids.append(ids)
        valid_boxes = [r['boxes'][ids].astype(np.float) for ids in valid_ids]
        
        covered_boxes = [np.zeros(boxes.shape[0], dtype=bool) for boxes in valid_boxes]
        for scale_i, chips in enumerate(all_chips):
            if chips.shape[0]>0:
                overlaps = ignore_overlaps(chips, valid_boxes[scale_i])
                cids, pids = np.where(overlaps > self.chip_gt_overlap)
                for ind, cid in enumerate(cids):
                    cur_chip = chips[cid]
                    cur_box = valid_boxes[scale_i][pids[ind]]
                    x1, x2, y1, y2 = max(cur_chip[0], cur_box[0]), min(cur_chip[2], cur_box[2]), \
                                     max(cur_chip[1], cur_box[1]), min(cur_chip[3], cur_box[3])
                    area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
                    if scale_i == len(self.scales) - 1:
                        # The coarsest scale (or the only scale)
                        if x2 - x1 >= 1 and y2 - y1 >= 1 and area >= self.valid_ranges[scale_i][0]:
                            props_in_chips[all_chip_ids[scale_i][cid]].append(valid_ids[scale_i][pids[ind]])
                            covered_boxes[scale_i][pids[ind]] = True
                    else:
                        if x2 - x1 >= 1 and y2 - y1 >= 1 and area <= self.valid_ranges[scale_i][1] and \
                                                          area >= self.valid_ranges[scale_i][0]:
                            props_in_chips[all_chip_ids[scale_i][cid]].append(valid_ids[scale_i][pids[ind]])
                            covered_boxes[scale_i][pids[ind]] = True

        if self.use_neg_chips:
            # ** Generate negative chips based on remaining boxes
            rem_valid_boxes = [valid_boxes[i][np.where(covered_boxes[i] == False)[0]] for i in range(len(self.scales))]
            neg_chips = []
            neg_props_in_chips = []
            first_available_chip_id = 0
            neg_chip_ids = []
            for scale_i, im_scale in enumerate(self.scales):
                chips = self.chip_generator.generate(rem_valid_boxes[scale_i] * im_scale, int(r['width'] * im_scale),
                                            int(r['height'] * im_scale), self.chip_size)
                neg_chips.append(np.array(chips, dtype=np.float) / im_scale)
                neg_props_in_chips += [[] for _ in chips]
                neg_chip_ids.append(np.arange(first_available_chip_id,first_available_chip_id+len(chips)))
                first_available_chip_id += len(chips)

            # ** Assign remaining boxes to negative chips based on max overlap
            neg_ids = [valid_ids[i][np.where(covered_boxes[i] == False)[0]] for i in range(len(self.scales))]
            for scale_i in range(len(self.scales)):
                if neg_chips[scale_i].shape[0]>0:
                    overlaps = ignore_overlaps(neg_chips[scale_i], rem_valid_boxes[scale_i])
                    max_ids = overlaps.argmax(axis=0)
                    for pi, cid in enumerate(max_ids):
                        cur_chip = neg_chips[scale_i][cid]
                        cur_box = rem_valid_boxes[scale_i][pi]
                        x1, x2, y1, y2 = max(cur_chip[0], cur_box[0]), min(cur_chip[2], cur_box[2]), \
                                         max(cur_chip[1], cur_box[1]), min(cur_chip[3], cur_box[3])
                        area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
                        if scale_i == len(self.scales) - 1:
                            if x2 - x1 >= 1 and y2 - y1 >= 1 and area >= self.valid_ranges[scale_i][0]:
                                neg_props_in_chips[neg_chip_ids[scale_i][cid]].append(neg_ids[scale_i][pi])
                        else:
                            if x2 - x1 >= 1 and y2 - y1 >= 1 and area < self.valid_ranges[scale_i][1]:
                                neg_props_in_chips[neg_chip_ids[scale_i][cid]].append(neg_ids[scale_i][pi])
            # Final negative chips extracted:
            final_neg_chips = []
            # IDs of proposals which are valid inside each of the negative chips:
            final_neg_props_in_chips = []
            chip_counter = 0
            for scale_i, chips in enumerate(neg_chips):
                im_scale = self.scales[scale_i]
                for chip in chips:
                    if len(neg_props_in_chips[chip_counter]) > 25 or (
                                    len(neg_props_in_chips[chip_counter]) > 10 and scale_i != 0):
                        final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                        final_neg_chips.append(
                            [chip, im_scale, int(r['height'] * im_scale), int(r['width'] * im_scale)])
                        chip_counter += 1

            r['neg_chips'] = final_neg_chips
            r['neg_props_in_chips'] = final_neg_props_in_chips
        for j in range(len(props_in_chips)):
            props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)
        if self.use_neg_chips:
            return props_in_chips, final_neg_chips, final_neg_props_in_chips
        else:
            return props_in_chips


def chip_generate(roidb, chip_worker, cfg, n_neg_per_im=2):

    #--- for debug ---#
    # roidb = roidb[:100]
    #--- for debug ---#

    crop_idx = [0] * len(roidb)
    chip_worker.reset()
    pool = Pool(cfg.num_process)
    # Devide the dataset and  extract chips for each part
    n_per_part = int(math.ceil(len(roidb) / float(cfg.chips_db_parts)))
    chips = []
    for i in range(cfg.chips_db_parts):
        print('chip_extractor', i) 
        chips += pool.map(chip_worker.chip_extractor,
                               roidb[i*n_per_part:min((i+1)*n_per_part, len(roidb))])

    # ----------for debug ---------------#
    # chips = []
    # temp_ = 0
    # for r in roidb:
    #     temp_ += 1
    #     print 'roidb', temp_
    #     chips.append(chip_worker.chip_extractor(r))
    # ----------for debug ---------------#

    chip_count = 0
    for i, r in enumerate(roidb):
        cs = chips[i]
        chip_count += len(cs)
        r['crops'] = cs

    if cfg.use_neg_chips:
        all_props_in_chips = []
        for i in range(cfg.chips_db_parts):
            print 'box_assigner %s (neg chips: %s)' %(i,cfg.n_neg_per_im)
            all_props_in_chips += pool.map(chip_worker.box_assigner,
                                   roidb[i*n_per_part:min((i+1)*n_per_part, len(roidb))])
        print 'len of all_props_in_chips:', len(all_props_in_chips)

        for ps, cur_roidb in zip(all_props_in_chips, roidb):
            cur_roidb['neg_crops'] = ps[1]

    chipindex = []
    if cfg.use_neg_chips:
        # Append negative chips
        for i, r in enumerate(roidb):
            cs = r['neg_crops']
            if len(cs) > 0:
                sel_inds = np.arange(len(cs))
                if len(cs) > n_neg_per_im:
                    sel_inds = np.random.permutation(sel_inds)[0:n_neg_per_im]
                for ind in sel_inds:
                    chip_count = chip_count + 1
                    r['crops'].append(r['neg_crops'][ind])
            for j in range(len(r['crops'])):
                chipindex.append(i)
    else:
        for i, r in enumerate(roidb):
            for j in range(len(r['crops'])):
                chipindex.append(i)

    print('Total number of extracted chips: {}'.format(chip_count))

    return chipindex