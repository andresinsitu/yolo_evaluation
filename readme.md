# Project to evaluate and get metrics from yolo format

Calculate metrics in YOLO format without having to convert labels to COCO first

## Metrics implemented
- $PR$ curve
- $P$ curve
- $R$ curve
- $F1$ score
- $mAP_{0.5}$
## Metrics to implement
- Confusion matrix

## Label format
Usual YOLOv5-8 label format. 

For predictions: 
`<class> <x_center> <y_center> <width> <height> <confidence> ` 

For ground of truth:
`<class> <x_center> <y_center> <width> <height> ` 

Also the values are normalized 

## How to:
Execute main script with following arguments:
- --img_path : image folder path
- --gt_path : ground of truth labels folder path
- --pred_path : prediction labels folder path
- --out_path :  path where the curves will be saved

python main.py --img_path --gt_path --pred_path --out_path 







