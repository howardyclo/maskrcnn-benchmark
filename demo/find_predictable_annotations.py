import os
import argparse
import requests
import pickle
import numpy as np

from tqdm import tqdm
from io import BytesIO
from PIL import Image

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from pycocotools.coco import COCO

import sys
sys.path.append('/home/v-yuclo/cocoapi/PythonAPI/pycocotools/') # You may need to modify this path.
import mask as maskUtils

def build_opt(parser):
    data_opt = parser.add_argument_group('Data')
    data_opt.add_argument('-root_dir', type=str, default='/home/v-yuclo/coco_dataset/',
                          help='Root directory of "train2014" and "annotations_trainval2014". \
                          We assume images and annotations are placed at the same root folder. ')
    data_opt.add_argument('-type', type=str, default='train',
                          help='Which type of COCO dataset to use. One of ["train", "val"].')
    data_opt.add_argument('-year', type=str, default='2014',
                          help='Which year of COCO dataset to use. One of ["2014", "2017"].')
    
    model_opt = parser.add_argument_group('Model')
    model_opt.add_argument('-cfg_file', type=str, default='../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml',
                           help='Config file of pretrained model to predict bboxes. See model choices in "../configs/caffe2/".')
    model_opt.add_argument('-score_threshold', type=float, default=0.7,
                           help='Score threshold for filtering predictions.')
    model_opt.add_argument('-iou_threshold', type=float, default=0.5,
                           help='IOU threshold for filtering predictions.')
    
    infer_opt = parser.add_argument_group('Inference')
    infer_opt.add_argument('-gpu', type=str, default=-1,
                           help='Specify which GPU for inference. If not given, use CPU instead.')
    
    dev_opt = parser.add_argument_group('Development')
    dev_opt.add_argument('-debug', action='store_true',
                         help='Only run one sample for testing.')
    
    opt = parser.parse_args()
    return opt

def config(opt):    
    # Update the config options with the config file.
    cfg.merge_from_file(opt.cfg_file)

    # Set device.
    if opt.gpu == -1:
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def imread(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_model(opt):
    model = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=opt.score_threshold,
    )
    
    return model

def load_coco_dataset(opt):
    coco = COCO(os.path.join(opt.root_dir, f'annotations_trainval{opt.year}/annotations/instances_{opt.type}{opt.year}.json'))
    return coco

def get_predictable_annotation_ids(opt, anns, predictions):
    """ 
    Args: 
        - `anns`: COCO annotation of an image.
        - `predictions`: Predictions of an image from the pretrained model.
    """
    g = [g['bbox'] for g in anns] # ground-truth bboxes (xywh)
    d = predictions.bbox.cpu().numpy().copy() # detected bboxes (xyxy)
    
    if not len(g) or not len(d):
        return []
    
    d[:,[2,3]] -= d[:,[0,1]] # convert to (xywh)
    iscrowd = [int(g['iscrowd']) for g in anns]
    
    # Compute IOUs.
    ious = maskUtils.iou(d, g, iscrowd) # shape: (#d, #g)

    # Find predictable ground-truth bbox ids.
    predictable_ann_ids = ious.argmax(axis=-1)[ious.max(axis=-1) >= opt.iou_threshold]
    
    return predictable_ann_ids

def main(opt):

    print('Loading pretrained model...')
    model = load_model(opt)
        
    print('Loading COCO dataset...')
    coco = load_coco_dataset(opt)
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids) # dict objects
    
    new_anns = {}
    
    for img in tqdm(imgs):
        
        try:
            img_filename = img['file_name']
            # Read image.
            image = imread(os.path.join(opt.root_dir, f'{opt.type}{opt.year}', img_filename))

            # Load ground-truth annotation of the image.
            ann_ids = coco.getAnnIds(imgIds=[img['id']])
            anns = coco.loadAnns(ann_ids)
            
            # Predict.
            result, predictions = model.run_on_opencv_image(image, return_predictions=True)

            # Find predictable ground-truth bbox ids.
            predictable_ann_ids = get_predictable_annotation_ids(opt, anns, predictions)
            
            # Label it.
            for ann_id in range(len(anns)):
                anns[ann_id]['predictable'] = ann_id in predictable_ann_ids
                
            # Save labeled annotation.
            new_anns[img['id']] = anns
    
        except Exception as e:
            print(f"Image ID: {img['id']} occurs error.")
            print(e)
            raise
            
        if opt.debug:
            break
    
    model_name = os.path.basename(opt.cfg_file).split('.yaml')[0]
    print(f'Saving new annotations predict by "{model_name}"...')
    with open(f'coco_{opt.type}{opt.year}_{model_name}.pkl', 'wb') as file:
        pickle.dump(new_anns, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='find_predictable_annotations.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt = build_opt(parser)
    config(opt)
    main(opt)