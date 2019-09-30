import os
from easydict import EasyDict as edict

config = edict()

# specify task type
config.eval_task_stype = ['det', ]
config.cache_dir = './cache'

# dataset setting
config.dataset = edict()
config.dataset.predict_roidb = './result.pkl'
config.dataset.gt_roidb = '/opt/hdfs/user/he.huang/common/dataset/coco2017/roidbs/val2017_det_gt_roidb.pkl'
config.dataset.coco_format_json = {
    'det': '/opt/hdfs/user/he.huang//common/dataset/coco2017/annotations/instances_val2017.json'
}

config.dataset.gt_seglabellst_path = None
if 'seg' in config.eval_task_stype:
    assert os.path.isfile(config.dataset.gt_seglabellst_path)


# for test filter
config.filter_strategy = edict()
config.filter_strategy.remove_empty_boxes = False
config.filter_strategy.max_num_images = None

