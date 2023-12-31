import numpy as np
import pandas as pd



def iou_matrix(bboxes1, bboxes2):
    """
    Calculate IoU of 2 arrays of boxes

    Args:
        bboxes1: Array of shape (N,4)
        bboxes2: Array of shape (M,4)

    Returns:
        np.array of shape(N,M)
    """

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def match_predictions(pred_classes, true_classes, iou):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) for different IoU thresholds

    Args:
        pred_classes (np.array): Predicted class indices of shape(N,).
        true_classes (np.array): Target class indices of shape(M,).
        iou (np.array): An MxN array containing the pairwise IoU values for predictions and ground of truth

    Returns:
        (np.array): Correct array of shape(N,10) for 10 IoU thresholds.
    """

    iout = np.linspace(0.5,0.95,10) #thresholds

    correct = np.zeros((pred_classes.shape[0], iout.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class

    for i, threshold in enumerate(iout):
        matches = np.nonzero(iou >= threshold)  # IoU >= threshold and classes match
        matches = np.array(matches).T

        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct.astype(bool)