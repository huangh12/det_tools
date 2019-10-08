import os
from easydict import EasyDict as edict

config = edict()

# -- specify task type -- #
config.cache_dir = './cache'
config.eval_task_type = ['det', ]
config.rpn_rcnn_num_branch = 1
config.rpn_do_test = False
config.mask_binary_thresh = 0.5


# -- dataset setting -- #
config.dataset = edict()
config.dataset.predict_roidb = './result.pkl'
config.dataset.gt_roidb = '/home/users/he.huang/project/det_tools/eval_tool/coco2017val.pkl'
config.dataset.coco_format_json = {
    'det': '/opt/hdfs/user/he.huang//common/dataset/coco2017/annotations/instances_val2017.json'
}
# only being used to generate GT json file
clsid2clsname = {
    i:str(i) for i in range(1,81)
}
coco_cat_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
               22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
               43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
               62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
clsid2catid = {    
    i: coco_cat_id[i-1] for i in range(1,81)
}
config.dataset.task_to_cls = None
config.dataset.gt_seglabellst_path = None
if 'seg' in config.eval_task_type:
    assert os.path.isfile(config.dataset.gt_seglabellst_path)


# -- test filter -- #
config.filter_strategy = edict()
config.filter_strategy.remove_empty_boxes = False
config.filter_strategy.max_num_images = None

