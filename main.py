from utils.load import *
from utils.ops import *
from utils.metrics import *

from pathlib import Path
import argparse


def main(paths):

    img_path = paths.img_path
    gt_path = paths.gt_path
    pred_path = paths.pred_path
    out_path = paths.out_path

    pred_df, gt_df = load_labels(img_path, gt_path, pred_path) # Form dataframes from labels
    pred_array, gt_array = get_bboxes_from_df(pred_df), get_bboxes_from_df(gt_df) # Get boxes
    iou = iou_matrix(gt_array, pred_array) # Calculate IOU matrix
    pred_classes, true_classes = get_class_indices(pred_df), get_class_indices(gt_df) # Get arrays of classes
    correct_array = match_predictions(pred_classes, true_classes, iou) # Get array of correct predictions 
    conf = get_confidence(pred_df) # Get confidence array

    Path(out_path).mkdir(parents=True, exist_ok=True) # Create directory and parents if it doesnt exist

    ap_per_class(correct_array, # Calculate the metrics
        conf,
        pred_classes,
        true_classes,
        plot=True,
        on_plot=None,
        save_dir= out_path,
        prefix='')



def get_args(): # define args
    parser = argparse.ArgumentParser("Calculate metrics from YOLO format")
    parser.add_argument(
        "--img_path",
        type=str,
        help="Path to images folder",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        help="Path to ground of truth labels folder",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        help="Path to predicted labels(with confidence score) folder",
    )
    
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to save the metrics",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    paths = get_args() # define args
    main(paths)



