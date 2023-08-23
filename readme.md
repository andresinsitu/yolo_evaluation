# Project to evaluate and get metrics from yolo format

## Metrics to implement
- $PR$ curve
- $P$ curve
- $R$ curve
- $F1$ score
- $mAP_{50}$
- Confusion matrix

## Label format

Usual YOLOv5-8 label format. 

For predictions: 
`<class> <x_center> <y_center> <width> <height> <confidence> ` 

For ground of truth:
`<class> <x_center> <y_center> <width> <height> ` 

Also the values are normalized 



