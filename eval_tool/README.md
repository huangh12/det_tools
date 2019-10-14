MS-COCO evaluation example:

```
hdfs dfs -get hdfs://hobot-bigdata/user/he.huang/project/det_tools/eval_tool/example/coco/coco2017val.pkl .
hdfs dfs -get hdfs://hobot-bigdata/user/he.huang/project/det_tools/eval_tool/example/coco/result.pkl .
```

WiderFace evaluation example:

```
hdfs dfs -get hdfs://hobot-bigdata/user/he.huang/project/det_tools/eval_tool/example/wider/val_add_ig_as_mat.pkl .
hdfs dfs -get hdfs://hobot-bigdata/user/he.huang/project/det_tools/eval_tool/example/wider/result_050.pkl .
```

input: 

`config.dataset.predict_roidb`: required, must have **image**, **id**, **boxes**(with score), **classes** key-value items, at least.

`config.dataset.coco_format_json`: optional, if not provided, it will be generated from `gt_roidb.pkl`

`config.dataset.gt_roidb`: optional, if `config.dataset.coco_format_json` is missing, then `gt_roidb.pkl` is required. It should have **image**, **id**, **boxes**, **gt_classes**, **height**, **width**, **area**, **ignore**, **iscrowd** key-value items at least. Refer to demo_coco_to_roidb.py for example.

`clsid2clsname`: map the class id to class name, which is required when `config.dataset.coco_format_json` is missing.

`clsid2catid`: map the class id to category id, which is required when `config.dataset.coco_format_json` is missing.


---

A typical modeling pipeline for object detection task.

`various raw annotation file(given by data annotator)` --> `unified format gt_roidb.pkl` --> CNN (for training)

`various raw annotation file(given by data annotator)` --> `unified format gt_roidb.pkl` --> `coco-format json file` (for evaluation)
                                                                        

Specifically, 
1. Construct one-to-one mapping from class name to category id (catetory id is a identifier to distinguish from different classes).

2. Construct one-to-one mapping from class name to class id (>1), class id is the int number which is predicted by the model, e.g., CNN.
(Under almost all time, category id can be same with class id)

3. Based on 1\&2, convert the raw annotation files (usually given by data annotator) to `gt_roidb.pkl` file. The `gt_roidb.pkl` is a list of dict
which includes at least **image**(str name), **id**(identifier), **boxes**, **gt_classes**, **height**, **width**, **area**, **ignore**, **iscrowd** key-value items. `gt_roidb.pkl` can replace the raw annotation file completely and serves for training directly. Refer to demo_coco_to_roidb.py for example.

1. CNN predicts on the validation images and output a file names `result.pkl`, which is also a list of dict like `gt_roidb.pkl`.
For `result.pkl`, it must have **image**, **id**, **boxes**, **classes** key-value items. What's more, the **boxes** of `result.pkl` is a 
**n * 5** array (np.array([[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...,])) while `gt_roidb.pkl`'s is a **n * 4** array
(np.array([[x1,y1,x2,y2],[x1,y1,x2,y2],...,])).
