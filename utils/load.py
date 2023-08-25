import pandas as pd
import os
from PIL import Image
import csv
import numpy as np

#import torch


def load_labels(image_dir,gt_dir,pred_dir):
    """
    Function for loading the labels in two dataframes, one for gt(ground of truth) and one for pred(predicted)

    Args:
        image_dir: Path to image folder
        gt_dir: Path to ground truth labels
        pred_dir: Path to predicted labels
    """
    #pred_df = pd.DataFrame(columns=['filename', 'class', 'xmin','ymin','xmax','ymax','conf'])
    #gt_df = pd.DataFrame(columns=['filename', 'class','xmin','ymin','xmax','ymax'])
    pred_list=[]
    gt_list=[]

    for image in os.listdir(image_dir):
        if image.endswith(('png','jpg')):
            image_path = os.path.join(image_dir, image)
            w,h = get_shape(image_path)
            pred_file = join_path_txt(pred_dir, image)
            gt_file = join_path_txt(gt_dir,image)
            #fname_without = os.path.splitext(fname)[0]

            gt_batch_list = list_from_text(gt_file,w,h)
            if gt_batch_list is not None:
                for row in gt_batch_list:
                    gt_list.append(row)

            pred_batch_list = list_from_text(pred_file,w,h)
            if pred_batch_list is not None:
                for row in pred_batch_list:
                    pred_list.append(row) 

    pred_df = pd.DataFrame(pred_list, columns = ['filename','class','xmin','ymin','xmax','ymax','conf'])
    gt_df = pd.DataFrame(gt_list, columns = ['filename','class', 'xmin','ymin','xmax','ymax'])
 
    return pred_df, gt_df


def get_bboxes_from_df(df):
    """
    Gets the bboxes out of the dataframe as numpy_array of shape(N,4), being N the number of rows

    Args:
        df: Dataframe with following columns: 'filename','class','xmin','ymin','xmax','ymax', and in the pred_df also 'conf'

    Returns:
        np.array: Of dimensions NxM, containing the IoU
    """

    array = df[['xmin','ymin','xmax','ymax']].to_numpy()
    return array



def get_shape(filename):
    """
    Function that gets shape of an image without loading it

    Args:
        filename: path to image
    
    Returns:
        tuple: width, height of the image
    """
    img=Image.open(filename)
    return (img.size[0], img.size[1])


def join_path_txt(path1,fname):
    """
    Joins paths and changes extension to .txt
    """
    fname_without = os.path.splitext(fname)[0]
    relative_path = os.path.join(path1,fname_without)
    return relative_path + '.txt'


def get_filename(path_to_file):
    """
    Gets filename from path(no extension)
    """
    filename = os.path.basename(path_to_file)
    filename = os.path.splitext(filename)[0]
    return filename

def list_from_text(path_to_file,w,h):
    """
    makes df out of the label txt, also add filename and transforms coordinates
    then it returns that df in list format, so it can be appended
    """
    # try:

    if not os.path.exists(path_to_file):
        list_df = None

    else:
        l = np.loadtxt(path_to_file)

        if not l.shape[0]:
            list_df = None

        else:
        #print(f'shape in {l.shape}')
            if len(l.shape) == 1:
                l = l.reshape(1,-1)
            #print(f'reshape in {l.shape}')
            dw = l[:,3] / 2  # half-width
            dh = l[:,4] / 2  # half-height
            z = np.empty_like(l)
            z[:,0] = l[:,0].astype(int)
            z[:,1] = (l[:,1] - dw)*w #calculate coordinates from xywhn to xyxy
            z[:,2] = (l[:,2] - dh)*h
            z[:,3] = (l[:,1] + dw)*w
            z[:,4] = (l[:,2] + dh)*h
            if len(l[0]) == 6: # only predicted labels(with confidence) has len 6
                z[:,5] = l[:,5]

            df = pd.DataFrame(z)

            filename= get_filename(path_to_file) #gets filename without extension

            df.insert(0,'filename',[filename for i in range(z.shape[0]) ])
            list_df = df.values.tolist()

        # except:
        #    print('EXCEPTION')
        #    list_df = None
    return list_df




