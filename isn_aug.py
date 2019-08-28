#coding:utf-8
'''
Description: This file is to transform the input roidb.pkl into augmented roidb_isn.pkl.
             The central idea comes from the paper https://arxiv.org/abs/1908.07323

step 1: Obatin the statistics of the boxes distribution after RESIZE, due to the RESIZED images are the real input to CNN. 
        Essentially, the boxes distribution should be drawn from the TEST set. Because it is the characteristics 
        of the TEST set (e.g., scale distribution of objects) that the models should adapt to. 
        Especially when the train set and test set are not i.i.d, which usually happens.

step 2: To obtain the optimal anchor scale/ratio setting from the given data.
        This is realized by search the scale/ratio space such that as many GT boxes as possible
        can be matched, in a deterministic rule, e.g. match decided by IoU value.
        The optimal anchor ratio can directly refer to the ratio of given data boxes, while the optimal
        anchor scale is somewhat complimcated. Again, these should be drawn from the TEST set as well.
        However, recent research exhibits good results by guiding the anchor via feature.
        This is a topic worth exploring and could reduce the difficulty of designing anchor manually.

step 3: The former step 1 gives the distribution of boxes to be detect. 
        We then crop patches from the image with ISN according to the roidbs.pkl.
        However, there are more than one method to crop the image, e.g., by scale(sqrt(w*h)) or by ratio, max(w, h).
        During the label assignment, the ignore boxes should be labeled as -1, -2 -3 and so on, instead being all labeld as -1.
        Simply labeled all ingnored boxes as -1 will inevitably cause more false positive.




author: he.huang
'''

from easydict import EasyDict as edict
from utils.data_workers import chip_worker, chip_generate

config = edict()
config.num_classes = 1  # exclude the background
config.orig_roidb_path = '/opt/hdfs/user/he.huang/project/wider_face_2019/dataset/WiderFace2019/roidbs/train.pkl'
config.test_scale = (800, 1333)



# ----------------- step 1: get boxes stat --------------------
print('----------------- step 1: get boxes stat --------------------')
import cPickle
import numpy as np

class Stats:
    def __init__(self, seq, name):
        seq = sorted(seq)
        self.arr = np.array(seq, dtype=np.float32)
        self.size = self.arr.shape[0]
        self.name = name
    
    def analysis(self, ndiv):
        pstr = '-----stat of %s-----\n' %self.name
        pstr += 'min: %f\n' %self.arr.min()
        pstr += 'max: %f\n' %self.arr.max()
        pstr += 'mean: %f\n' %self.arr.mean()
        pstr += 'numbers: %f\n' %self.size
        ppart, npart = 100./ndiv, round((self.size-1)/ndiv)
        for i in range(ndiv):
            pstr += '%.2f%s %s %.2f%s : %.2f -> %.2f\n' \
                %(i*ppart, '%', '~', (i+1)*ppart, '%', self.arr[int(i*npart)], self.arr[int((i+1)*npart)])
        print(pstr)


def get_im_scale(size, im_shape):
    assert isinstance(size, tuple) or isinstance(size, list)
    target_size = size[0]
    max_size = size[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


boxClsStat = {}
boxes = []
boxRatioStat = []
boxScaleStat = []
rScaleStat = []
with open(config.orig_roidb_path, 'r') as f:
    orig_roidb = cPickle.load(f)


print('distribute the ignore label -1 to all %d classes (exclude the background!!)' %config.num_classes)
for r in orig_roidb:

    '''
    preprocess the gt_classes:
    In current annotation rule, the ignore region are simply labeled as -1.
    Since we want to generate class-aware ignore region, we manually distribute the -1 region to all classes    
    '''
    assert 0 not in r['gt_classes'], '0 in gt_classes!'
    ig_idx = np.where(r['gt_classes'] == -1)[0]
    append_cls = np.tile(-1 * (2 + np.arange(config.num_classes-1)), len(ig_idx)).astype(np.int32) 
    append_box = np.empty((0, 4), dtype=np.float32)
    for idx in ig_idx:
        append_box = np.vstack([append_box, np.tile(r['boxes'][idx], (1,config.num_classes-1))]).astype(np.float32)

    r['gt_classes'] = np.hstack( [r['gt_classes'], append_cls] )
    r['boxes'] = np.vstack( [r['boxes'], append_box] )
    assert r['gt_classes'].shape[0] == r['boxes'].shape[0]


    # do stat
    unique, counts = np.unique(r['gt_classes'], return_counts=True)
    for u, c in zip(unique, counts):
        if u not in boxClsStat:
            boxClsStat[u] = c
        else:
            boxClsStat[u] += c

    boxes.extend( (r['boxes'].tolist()) )
    boxes_ratio = (r['boxes'][:,3]-r['boxes'][:,1]) / (r['boxes'][:,2]-r['boxes'][:,0])
    boxes_scale = np.sqrt((r['boxes'][:,3]-r['boxes'][:,1]) * (r['boxes'][:,2]-r['boxes'][:,0]))
    boxRatioStat.extend( (boxes_ratio).tolist() )
    boxScaleStat.extend( (boxes_scale).tolist() )

    # We assume the r['height'] and r['width'] are correct, which should be checked first.
    im_scale = get_im_scale(size=config.test_scale, im_shape=(r['height'],r['width']))
    rScaleStat.extend( (im_scale*boxes_scale).tolist() )

boxRatioStat_ = Stats(boxRatioStat, name='boxRatioStat')
boxScaleStat_ = Stats(boxScaleStat, name='boxScaleStat')
boxRatioStat_.analysis(ndiv=3)
boxScaleStat_.analysis(ndiv=10)



# ----------------- step 2: ISN --------------------
import copy
from utils.bbox.bbox_transform import *
import cv2
import os

print('----------------- step 2: ISN --------------------')
pstr = 'Since the image height/width is utilized in the ISN process, it\'s highly\n'+\
       'encouraged that you check the roidb first, such that the height and width in the\n'+\
       'each roidb is strictly correct!\n'
print(pstr)


def draw_boxes(im, boxes, gt_classes):
    for i, box in enumerate(boxes):
        color = (0,0,255) if gt_classes[i]<0 else (0,255,0)
        cv2.rectangle(im, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])),color,1)
        area = np.sqrt((box[2]-box[0])*(box[3]-box[1]))
        cv2.putText(im, "%d/%.2f" % (gt_classes[i], area), (int(box[2]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return im

def get_valid_inds(boxes, shape, scale_range, chip_gt_overlap):
    h = shape[0]
    w = shape[1]
    clip_boxes = np.zeros_like(boxes, dtype=np.float32)
    clip_boxes[:,0::2] = np.clip(boxes[:,0::2], 0, w-1)
    clip_boxes[:,1::2] = np.clip(boxes[:,1::2], 0, h-1)
    area = np.sqrt((clip_boxes[:,3]-clip_boxes[:,1])*(clip_boxes[:,2]-clip_boxes[:,0]))
    valid_ind_1 = np.where( (area>=scale_range[0]) & (area<=scale_range[1]) )[0]
    overlap = ignore_overlaps(np.array([[0,0,w,h]]).astype(np.float), boxes)
    valid_ind_2 = np.where( overlap > chip_gt_overlap )[1]
    return np.intersect1d(valid_ind_1, valid_ind_2)

# the gt boxes which are invalid in the chip but have an overlap 
# more than chip_truncation_ignore are marked ignored region in the chip
config.chip_gt_overlap = 0.7
config.chip_truncation_ignore = 0.4
## CHIP GENERATION PARAMS
config.chip_size = (576, 576)
# config.chip_size = [640,640]
# Whether to use C++ or python code for chip generation
config.cpp_chips = True
# Multi-processing params
# These parameters are used for parallel chip generation, NMS, etc.
# Please consider adjusting them for your system
config.num_process = 16
# How many parts the dataset should be divided to for parallel chip generation
# This is used to keep the memory limited
config.chips_db_parts = 10
config.use_neg_chips = True
config.n_neg_per_im = 3
config.pyramid_scales = [4.0, 2.0, 1.0, 0.5, 0.25]
config.chip_stride = np.random.randint(56, 60)  
# config.valid_ranges = [[-1,80], [8,160], [16,320], [32,640], [64,-1]]
# fpn-v-5 cs576
config.valid_ranges = [[-1,140], [8,280], [16,560], [32,1120], [64,-1]]
# fpn-v-5_1 cs576
# config.valid_ranges = [[-1,140], [16,280], [32,560], [64,1120], [128,-1]]
# all
# config.valid_ranges = [[0,1000], [0,1000], [0,1000], [0,1000], [0,1000]]

# the setting of the learning range should refer to the distribution characteristics of TEST set.
config.learning_range = config.valid_ranges[config.pyramid_scales.index(1)] 
config.isn_roidb_save_path = './train_isn.pkl'
# ----------

# get the chip
chip_worker_ = chip_worker(chip_size=config.chip_size[0], cfg=config)
chip_index = chip_generate(orig_roidb, chip_worker_, cfg=config, n_neg_per_im=config.n_neg_per_im)

# distribute the chip into isn_roidb
isn_roidb = []
for r in orig_roidb:
    for crop in r['crops']:
        append_ids = np.zeros([0], dtype=np.int32)
        crop_coordinates, im_scale, _, _ = crop
        all_boxes = copy.deepcopy(r['boxes']).astype(np.float)
        # ----------------- for debug-------------------#
        # img_dir = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/all_train_data/train_images/'
        # im_ori = cv2.imread(os.path.join(img_dir, r['image']), cv2.IMREAD_COLOR)
        # im_ori_draw = draw_boxes(copy.deepcopy(im_ori), r['boxes'], r['gt_classes'])

        # img_name = os.path.splitext(r['image'])[0]
        # path_ = 'img_chip_check/'
        # if not os.path.exists(path_):
        #     os.makedirs(path_)
        # cv2.imwrite('{}/{}.jpg'.format(path_, img_name),im_ori_draw)
        #----------------- for debug-------------------#
        
        # shifts and scale all boxes and get patch size
        all_boxes = (all_boxes - crop_coordinates[[0,1,0,1]]) * im_scale
        x1,y1,x2,y2 = crop_coordinates
        patch_h = int( (min(r['height'], y2) - y1) * im_scale )
        patch_w = int( (min(r['width'], x2) - x1) * im_scale )

        # valid 
        valid_inds = get_valid_inds(all_boxes, (patch_h,patch_w), config.learning_range, config.chip_gt_overlap)
        valid_boxes = all_boxes[valid_inds]
        valid_gt_classes = r['gt_classes'][valid_inds]

        # invalid (ignore the region of invalid gt boxes with large overlap with current chip)
        invalid_inds = np.array(list(set(range(len(r['gt_classes']))) - set(valid_inds))).astype(np.int)
        invalid_boxes = all_boxes[invalid_inds]
        invalid_gt_classes = r['gt_classes'][invalid_inds]

        # the invalid that should be ignored
        overlap = ignore_overlaps(np.array([[0,0,patch_w,patch_h]]).astype(np.float), invalid_boxes)
        append_ignore_inds = np.where(overlap > config.chip_truncation_ignore)[1]
        ignore_gt_classes = copy.deepcopy(invalid_gt_classes[append_ignore_inds])
        change_idx = np.where(ignore_gt_classes > 0)[0]
        ignore_gt_classes[change_idx] = -1 * ignore_gt_classes[change_idx]

        # the final boxes and gt_classes in the croped patch
        gt_boxes = np.vstack([valid_boxes, invalid_boxes[append_ignore_inds]]).astype(np.float32)
        gt_classes = np.hstack([valid_gt_classes, ignore_gt_classes]).astype(np.int32)
        assert gt_boxes.shape[0] == gt_classes.shape[0]

        # clip the gt boxes
        gt_boxes[:,0] = np.clip(gt_boxes[:,0], 0, patch_w-1)
        gt_boxes[:,1] = np.clip(gt_boxes[:,1], 0, patch_h-1)
        gt_boxes[:,2] = np.clip(gt_boxes[:,2], 0, patch_w-1)
        gt_boxes[:,3] = np.clip(gt_boxes[:,3], 0, patch_h-1)

        crop_roidb = dict()
        crop_roidb['image'] = r['image']
        crop_roidb['boxes'] = gt_boxes
        crop_roidb['gt_classes'] = gt_classes
        crop_roidb['crop'] = np.append(crop_coordinates, im_scale)
        isn_roidb.append(crop_roidb)
        #----------------- for debug-------------------#
        # assert len(crop_coordinates)==4
        # x1,y1,x2,y2 = crop_coordinates
        # img = im_ori[int(y1):int(y2),int(x1):int(x2),:]
        # img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        # im_draw = draw_boxes(img, gt_boxes, gt_classes)
        # import datetime
        # chip_name = os.path.splitext(r['image'])[0] + datetime.datetime.now().strftime('%H-%M-%S')
        # cv2.imwrite('{}/{}.jpg'.format(path_, chip_name), im_draw)
        # print('shape {} --> {}'.format(im_ori.shape, img.shape)  )
        # import pdb; pdb.set_trace()
        #----------------- for debug-------------------#

with open(config.isn_roidb_save_path, 'w') as f:
    cPickle.dump(isn_roidb, f, cPickle.HIGHEST_PROTOCOL)
    print('Dump generated isn_roidb to %s' %config.isn_roidb_save_path)

