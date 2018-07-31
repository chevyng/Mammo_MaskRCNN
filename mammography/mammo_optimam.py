"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
    python nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
    python nucleus.py train --dataset="C:/Users/Chevy/Documents/GitHub/Mammo_MaskRCNN/datasets/nucleus/stage1_train" --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class MammoConfig(Config):
    """Configuration for training on the mammography segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "mammo"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + mass

    # Number of training and validation steps per epoch
    # val_length = 238
    # STEPS_PER_EPOCH = (657 - val_length) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, val_length // IMAGES_PER_GPU)
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # STEPS_PER_EPOCH = (1229 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between mass and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    # IMAGE_RESIZE_MODE = "crop"
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 3000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.9
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([54.78, 54.78, 54.78])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class MammoInferenceConfig(MammoConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class MammoDataset(utils.Dataset):

    def load_metadata(self, dataset_dir, file_name, sheet_name, col_name):
        # dataset_dir = "/home/chevy/Desktop/Github/Mammo_MaskRCNN/datasets/mammo"
        # file_name = "images_lesions_all.xlsx"
        file_dir = os.path.join(dataset_dir, file_name)
        df = pd.read_excel(file_dir, sheet_name=sheet_name)
        image_ids = df[col_name]
        return image_ids



    def load_mammo(self, dataset_dir, subset, isOptimam=False, json_filename='mammo_normal.json'):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "mass_train", "mass_test", "calc_test",
                          "optimam_train", "optimam_val"]
        # subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        # subset_dir = "mass_train" if subset in ["mass_train", "val"] else subset
        subset_dir = "optimam_train" if subset in ["train", "optimam_train", "optimam_val"] else subset
        # subset_dir = subset
        json_dir = dataset_dir
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        print(subset)
        if not isOptimam:
            image_ids = next(os.walk(dataset_dir))[2]
        else:
            json_path = os.path.join(json_dir, json_filename)
            annotations = json.load(open(json_path))
            images = annotations['images']
            annos = annotations['annotations']
            annos_ids = []
            for i in range(len(annos)):
                annos_ids.append(annos[i]['image_id'])

            image_ids = list(set(annos_ids))
            annos_ids_dict = {k:[] for k in image_ids}
            images_ids_dict = {k:[] for k in image_ids}
            idx = 0
            for name in annos_ids:
                annos_ids_dict[name].append(idx)
                idx += 1

            idx = 0
            for d in images:
                images_ids_dict[d['id']] = idx
                idx += 1

            d = {k:{} for k in image_ids}
            for name in annos_ids:
                anno_idxs = annos_ids_dict[name]
                img_idx = images_ids_dict[name]
                if len(anno_idxs) == 1:
                    d[name]['image_id'] = name
                    d[name]['file_name'] = images[img_idx]['file_name']
                    d[name]['height'] = images[img_idx]['height']
                    d[name]['width'] = images[img_idx]['width']
                    d[name]['bbox'] = [tuple(annos[anno_idxs[0]]['bbox'])]
                    d[name]['iscrowd'] = annos[anno_idxs[0]]['iscrowd']
                    d[name]['category_id'] = [annos[anno_idxs[0]]['category_id']]
                    d[name]['id'] = annos[anno_idxs[0]]['id']
                else:
                    d[name]['image_id'] = name
                    d[name]['file_name'] = images[img_idx]['file_name']
                    d[name]['height'] = images[img_idx]['height']
                    d[name]['width'] = images[img_idx]['width']
                    d[name]['bbox'] = []
                    d[name]['category_id'] = []
                    d[name]['id'] = []
                    for idx in anno_idxs:
                        d[name]['bbox'].append(tuple(annos[idx]['bbox']))
                        d[name]['category_id'].append(annos[idx]['category_id'])
                        d[name]['id'].append(annos[idx]['id'])



        val_size = round(len(image_ids)* 0.20)
        np.random.seed(0)
        np.random.shuffle(image_ids)
        VAL_IMAGE_IDS = image_ids[:val_size]

        if subset == "val" or subset == "optimam_val":
            image_ids = VAL_IMAGE_IDS
        else:
            if subset == "train" or subset == "mass_train" or subset == "optimam_train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add classes. We have three class.
        # Naming the dataset 'optimam', and the 3 classes
        if isOptimam:
            self.add_class("optimam", 1, "MALIGNANT")
            self.add_class("optimam", 2, "BENIGN")
            self.add_class("optimam", 3, "NORMAL")
            for image_id in image_ids:
                image_filename = d[image_id]['file_name']
                image_path = os.path.join(dataset_dir, image_filename)
                self.add_image(
                    "optimam",
                    image_id=image_id,
                    path=image_path,
                    height = d[image_id]['height'],
                    width = d[image_id]['width'],
                    bbox = d[image_id]['bbox'],
                    catId = d[image_id]['category_id']
                    )
        else:
            self.add_class("mass", 1, "mass")
            for image_id in image_ids:
                self.add_image(
                    "mass",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id))

        # Add images

        return image_ids

    def load_bbox(self, image_ids, loaded_image_ids):
        # Get a list of boolean where only images in optimam folder is true so
        # that we can filter the entire list for images we have only
        loaded_image_present = image_ids['ImageSOPIUID'].isin(loaded_image_ids)
        img_here = image_ids[loaded_image_present]
        loaded_images = {k:[] for k in loaded_image_ids}

        for name in img_here['ImageSOPIUID']:
            ## Index is required because some images have more than 1 bbox coordinate
            ## boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
            idx = len(loaded_images[name])
        #     d[name][0] += 1
            x1 = img_here.loc[img_here['ImageSOPIUID'] == name, 'X1'].iloc[idx]
            x2 = img_here.loc[img_here['ImageSOPIUID'] == name, 'X2'].iloc[idx]
            y1 = img_here.loc[img_here['ImageSOPIUID'] == name, 'Y1'].iloc[idx]
            y2 = img_here.loc[img_here['ImageSOPIUID'] == name, 'Y2'].iloc[idx]
            loaded_images[name].append((y1,x1,y2,x2))


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mass":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MammoDataset()
    dataset_train.load_mammo(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MammoDataset()
    dataset_val.load_mammo(dataset_dir, "optimam_val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    # augmentation = iaa.SomeOf((2, 3), [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5)
    # ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=None,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=None,
                layers='all')

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MammoConfig()
    else:
        config = MammoInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
