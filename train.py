import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
import itertools

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from config_utils import read_config
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.data import DatasetCatalog, MetadataCatalog

import random
from detectron2.utils.visualizer import Visualizer

from detectron2.config import get_cfg


# write a function that loads the dataset into detectron2's standard format
def get_microcontroller_dicts(csv_file, img_dir):
    df = pd.read_csv(csv_file)
    df['filename'] = df['filename'].map(lambda x: img_dir+x)


    df['class_int'] = df['class'].map(lambda x: classes.index(x))

    dataset_dicts = []
    for filename in df['filename'].unique().tolist():
        record = {}
        
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        objs = []
        for index, row in df[(df['filename']==filename)].iterrows():
          obj= {
              'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
              'bbox_mode': BoxMode.XYXY_ABS,
              'category_id': row['class_int'],
              "iscrowd": 0
          }
          objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((800,600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def register_data():

    for d in ["train", "valid"]:
        DatasetCatalog.register(data_path + d, lambda d=d: get_microcontroller_dicts(data_path + d + '_labels.csv', data_path + d+'/'))
        MetadataCatalog.get(data_path + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get(data_path + '/train')

def show_training_images():
    dataset_dicts = DatasetCatalog.get(data_path + '/train')
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)
        v = v.draw_dataset_dict(d)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

def _create_cfg(cfg, model_name, train_dir, ckpt, batsh_size, max_iterations, num_classes, workers=2):
    cfg.merge_from_file(model_zoo.get_config_file(model_name)) #"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.DATASETS.TRAIN = (train_dir)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ckpt_path)#"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = max_iterations
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

if __name__=="__main__":
    import os
    print("Pytorch version: {}, GPU detected: {}".format(torch.__version__, torch.cuda.is_available()))
    data=read_config("config.yaml")
    data_path= data['data_path']
    classes = data['names']
    print(data)
    print('registrering data')
    for d in ["train", "valid"]:
      DatasetCatalog.register('./data/' + d, lambda d=d: get_microcontroller_dicts('./data/' + d + '_labels.csv', './data/' + d+'/'))
      MetadataCatalog.get('./data/' + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get('./data/train')
    print("Visualizing training data")
    #show_training_images()
    model_name=data['model_name']
    train_dir=os.path.join(data['data_path'], 'train')
    ckpt_path= data['model_ckpt']
    batch_size= data['batch_size']
    max_iterations=data['max_iterations']

    cfg = get_cfg()


    _create_cfg(cfg, model_name, train_dir, ckpt=data['model_ckpt'], batsh_size=batch_size, max_iterations=max_iterations,num_classes=len(classes))
    print('model configuration has been donne succefully')

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()







    







