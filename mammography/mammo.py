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

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = ['P_01516_RIGHT_MLO', 'P_00626_LEFT_CC', 'P_00265_RIGHT_CC', 'P_00803_LEFT_CC',
         'P_00509_RIGHT_CC', 'P_00648_LEFT_MLO', 'P_00363_LEFT_CC', 'P_01543_RIGHT_MLO', 'P_00451_LEFT_MLO',
         'P_01317_RIGHT_MLO', 'P_00611_RIGHT_MLO', 'P_00670_RIGHT_MLO', 'P_00215_RIGHT_CC', 'P_01749_LEFT_MLO',
         'P_01821_LEFT_CC', 'P_01702_RIGHT_MLO', 'P_01388_LEFT_MLO', 'P_01079_RIGHT_MLO', 'P_00915_RIGHT_CC',
         'P_01499_RIGHT_CC', 'P_00462_LEFT_MLO', 'P_01084_LEFT_MLO', 'P_01426_RIGHT_CC', 'P_00506_LEFT_MLO',
         'P_01270_RIGHT_CC', 'P_01355_LEFT_MLO', 'P_00841_RIGHT_MLO', 'P_00952_RIGHT_CC', 'P_00226_LEFT_MLO',
         'P_00027_RIGHT_CC', 'P_01540_LEFT_MLO', 'P_01832_RIGHT_MLO', 'P_00396_LEFT_CC', 'P_01008_RIGHT_CC',
         'P_00106_RIGHT_MLO', 'P_01243_LEFT_MLO', 'P_00539_RIGHT_CC', 'P_01720_RIGHT_MLO', 'P_00314_RIGHT_MLO',
         'P_01131_LEFT_MLO', 'P_01475_LEFT_CC', 'P_01158_RIGHT_CC', 'P_00984_RIGHT_MLO', 'P_01232_RIGHT_CC',
         'P_01635_RIGHT_CC', 'P_01330_RIGHT_MLO', 'P_01864_LEFT_CC', 'P_01491_RIGHT_MLO', 'P_00660_LEFT_MLO',
         'P_01564_RIGHT_CC', 'P_00226_LEFT_CC', 'P_01654_RIGHT_MLO', 'P_01646_LEFT_MLO', 'P_01780_LEFT_MLO',
         'P_01427_LEFT_MLO', 'P_00148_RIGHT_CC', 'P_00554_LEFT_CC', 'P_00823_RIGHT_CC', 'P_01120_LEFT_MLO',
         'P_00160_RIGHT_CC', 'P_01398_LEFT_CC', 'P_00450_LEFT_MLO', 'P_00004_LEFT_CC', 'P_01413_RIGHT_CC',
         'P_01035_RIGHT_CC', 'P_01613_LEFT_MLO', 'P_01383_LEFT_CC', 'P_01394_RIGHT_CC', 'P_01876_LEFT_CC',
         'P_00931_LEFT_MLO', 'P_01302_RIGHT_CC', 'P_00731_RIGHT_MLO', 'P_00976_LEFT_MLO', 'P_01243_LEFT_CC',
         'P_00929_LEFT_MLO', 'P_00939_RIGHT_MLO', 'P_00081_RIGHT_MLO', 'P_01262_RIGHT_CC', 'P_00778_RIGHT_CC',
         'P_01506_LEFT_CC', 'P_01583_RIGHT_MLO', 'P_01362_LEFT_MLO', 'P_00611_RIGHT_CC', 'P_01822_LEFT_MLO',
         'P_00913_LEFT_MLO', 'P_00122_RIGHT_MLO', 'P_00797_LEFT_CC', 'P_00119_LEFT_MLO', 'P_01754_RIGHT_MLO',
         'P_01179_LEFT_CC', 'P_01761_LEFT_CC', 'P_01165_LEFT_MLO', 'P_01009_RIGHT_MLO', 'P_00665_LEFT_MLO',
         'P_01642_RIGHT_MLO', 'P_01720_RIGHT_CC', 'P_00437_LEFT_CC', 'P_00764_RIGHT_MLO', 'P_00021_LEFT_CC',
         'P_01439_LEFT_MLO', 'P_01356_LEFT_CC', 'P_01175_LEFT_MLO', 'P_00499_RIGHT_MLO', 'P_01230_LEFT_MLO',
         'P_01503_LEFT_CC', 'P_01698_RIGHT_CC', 'P_01052_LEFT_MLO', 'P_01710_LEFT_MLO', 'P_00080_RIGHT_MLO',
         'P_00526_RIGHT_MLO', 'P_01557_RIGHT_CC', 'P_00584_LEFT_MLO', 'P_01461_RIGHT_MLO', 'P_00004_RIGHT_MLO',
         'P_00931_LEFT_CC', 'P_00640_RIGHT_MLO', 'P_01181_LEFT_CC', 'P_01340_LEFT_CC', 'P_00651_RIGHT_MLO',
         'P_00096_RIGHT_CC', 'P_01411_RIGHT_MLO', 'P_01642_RIGHT_CC', 'P_00816_RIGHT_CC', 'P_01606_RIGHT_MLO',
         'P_00018_RIGHT_CC', 'P_01301_RIGHT_MLO', 'P_01635_RIGHT_MLO', 'P_00086_RIGHT_MLO', 'P_00739_LEFT_CC',
         'P_01264_LEFT_MLO', 'P_00169_RIGHT_MLO', 'P_00626_LEFT_MLO', 'P_01447_RIGHT_MLO', 'P_01175_RIGHT_CC',
         'P_01332_RIGHT_CC', 'P_00065_LEFT_MLO', 'P_01267_RIGHT_MLO', 'P_00309_LEFT_CC', 'P_00706_RIGHT_MLO',
         'P_00694_RIGHT_MLO', 'P_00149_LEFT_MLO', 'P_00775_LEFT_MLO', 'P_01034_RIGHT_MLO', 'P_01238_RIGHT_CC',
         'P_01268_RIGHT_MLO', 'P_01130_RIGHT_CC', 'P_01745_RIGHT_MLO', 'P_00634_LEFT_MLO', 'P_00604_LEFT_CC',
         'P_01379_LEFT_MLO', 'P_00534_LEFT_CC', 'P_00074_LEFT_MLO', 'P_00419_RIGHT_CC', 'P_00207_LEFT_MLO',
         'P_00287_RIGHT_CC', 'P_00092_LEFT_MLO', 'P_00515_RIGHT_CC', 'P_00432_LEFT_MLO', 'P_00109_LEFT_CC',
         'P_01887_LEFT_CC', 'P_02092_LEFT_CC', 'P_01185_RIGHT_MLO', 'P_01035_RIGHT_MLO', 'P_01326_LEFT_CC',
         'P_00079_RIGHT_CC', 'P_00742_LEFT_MLO', 'P_00698_RIGHT_CC', 'P_01265_RIGHT_CC', 'P_00995_LEFT_CC',
         'P_00700_RIGHT_MLO', 'P_01097_LEFT_MLO', 'P_01556_LEFT_CC', 'P_01866_RIGHT_CC', 'P_00604_LEFT_MLO',
         'P_01034_RIGHT_CC', 'P_01860_RIGHT_CC', 'P_01207_RIGHT_MLO', 'P_01430_RIGHT_MLO', 'P_01532_LEFT_CC',
         'P_00577_RIGHT_CC', 'P_01418_RIGHT_MLO', 'P_00648_LEFT_CC', 'P_00698_RIGHT_MLO', 'P_00581_LEFT_MLO',
         'P_01164_RIGHT_MLO', 'P_01946_RIGHT_MLO', 'P_01712_LEFT_CC', 'P_01124_RIGHT_MLO', 'P_00690_LEFT_MLO',
         'P_00487_RIGHT_CC', 'P_00702_RIGHT_CC', 'P_00572_RIGHT_CC', 'P_00417_RIGHT_CC', 'P_00044_RIGHT_CC',
         'P_01717_LEFT_MLO', 'P_01475_LEFT_MLO', 'P_01600_RIGHT_CC', 'P_00332_LEFT_CC', 'P_00577_RIGHT_MLO',
         'P_00779_LEFT_MLO', 'P_00801_LEFT_MLO', 'P_00836_LEFT_CC', 'P_00376_RIGHT_CC', 'P_01701_LEFT_CC',
         'P_00810_RIGHT_MLO', 'P_00225_RIGHT_MLO', 'P_00199_LEFT_MLO', 'P_01173_RIGHT_CC', 'P_01560_RIGHT_CC',
         'P_01850_RIGHT_MLO', 'P_00279_LEFT_CC', 'P_00711_LEFT_MLO', 'P_00487_RIGHT_MLO', 'P_00896_LEFT_CC',
         'P_01373_RIGHT_CC', 'P_01302_RIGHT_MLO', 'P_01686_RIGHT_CC', 'P_00224_RIGHT_MLO', 'P_01394_LEFT_MLO',
         'P_01486_RIGHT_CC', 'P_00917_RIGHT_MLO', 'P_00958_LEFT_CC', 'P_01641_LEFT_CC', 'P_00419_RIGHT_MLO',
         'P_01057_RIGHT_CC', 'P_00802_LEFT_CC', 'P_00747_LEFT_CC', 'P_01047_LEFT_CC', 'P_00041_LEFT_MLO',
         'P_01841_RIGHT_MLO', 'P_00586_LEFT_MLO', 'P_01788_RIGHT_CC', 'P_01620_RIGHT_CC', 'P_01632_RIGHT_MLO',
         'P_00717_RIGHT_MLO', 'P_01499_RIGHT_MLO', 'P_01685_RIGHT_MLO', 'P_01638_RIGHT_MLO', 'P_01305_LEFT_CC',
         'P_01663_RIGHT_CC', 'P_00972_LEFT_MLO', 'P_01883_LEFT_CC', 'P_00978_RIGHT_MLO', 'P_01753_RIGHT_CC', 'P_01155_RIGHT_MLO',
         'P_01049_LEFT_MLO', 'P_01838_LEFT_MLO', 'P_00664_RIGHT_CC', 'P_00302_LEFT_CC', 'P_02195_RIGHT_MLO', 'P_01784_LEFT_CC',
         'P_01169_LEFT_MLO', 'P_01497_LEFT_MLO', 'P_01314_LEFT_MLO', 'P_01156_LEFT_CC', 'P_02348_RIGHT_MLO', 'P_01184_LEFT_MLO',
         'P_00639_RIGHT_MLO', 'P_01087_LEFT_CC', 'P_01522_RIGHT_CC', 'P_00060_RIGHT_CC', 'P_00674_RIGHT_MLO', 'P_00552_RIGHT_MLO',
         'P_00606_LEFT_CC', 'P_01126_LEFT_CC', 'P_01296_RIGHT_MLO', 'P_01885_LEFT_CC', 'P_01400_LEFT_MLO', 'P_01276_LEFT_CC',
         'P_00652_LEFT_CC', 'P_00043_LEFT_MLO', 'P_00047_LEFT_CC', 'P_01219_LEFT_MLO', 'P_01709_RIGHT_MLO', 'P_00432_RIGHT_MLO',
         'P_00196_RIGHT_MLO', 'P_00031_LEFT_CC', 'P_01838_RIGHT_CC', 'P_02270_RIGHT_MLO', 'P_01585_RIGHT_MLO', 'P_00701_LEFT_CC',
         'P_00293_LEFT_MLO', 'P_01129_RIGHT_CC', 'P_00539_RIGHT_CC', 'P_00321_LEFT_MLO', 'P_00991_LEFT_MLO', 'P_02115_LEFT_CC',
         'P_00048_RIGHT_MLO', 'P_01065_LEFT_CC', 'P_01143_RIGHT_CC', 'P_01033_LEFT_MLO', 'P_00467_LEFT_MLO', 'P_01830_RIGHT_CC',
         'P_00912_LEFT_CC', 'P_00854_RIGHT_CC', 'P_01169_LEFT_CC', 'P_01262_RIGHT_MLO', 'P_00792_LEFT_MLO', 'P_00744_RIGHT_MLO',
         'P_02522_RIGHT_MLO', 'P_02115_LEFT_MLO', 'P_01256_RIGHT_CC', 'P_00251_RIGHT_MLO', 'P_00360_RIGHT_CC', 'P_00819_RIGHT_MLO',
         'P_01454_LEFT_MLO', 'P_01234_RIGHT_CC', 'P_00582_RIGHT_MLO', 'P_02508_LEFT_CC', 'P_00557_RIGHT_MLO', 'P_00933_LEFT_CC',
         'P_01056_RIGHT_MLO', 'P_00710_RIGHT_CC', 'P_01274_RIGHT_MLO', 'P_00555_RIGHT_CC', 'P_00808_LEFT_MLO', 'P_01369_RIGHT_MLO',
         'P_01161_LEFT_MLO', 'P_00945_RIGHT_MLO', 'P_01136_RIGHT_MLO', 'P_01172_LEFT_CC', 'P_00993_RIGHT_CC', 'P_01276_RIGHT_CC',
         'P_00270_LEFT_CC', 'P_00621_LEFT_CC', 'P_01170_RIGHT_MLO', 'P_02572_LEFT_CC', 'P_01664_LEFT_MLO', 'P_01188_RIGHT_CC',
         'P_02402_RIGHT_MLO', 'P_01455_RIGHT_CC', 'P_01108_LEFT_MLO', 'P_01160_RIGHT_MLO', 'P_00701_LEFT_MLO', 'P_01571_RIGHT_CC',
         'P_00631_RIGHT_MLO', 'P_00445_LEFT_MLO', 'P_01579_RIGHT_MLO', 'P_00955_RIGHT_CC', 'P_01406_RIGHT_MLO', 'P_01455_RIGHT_MLO',
         'P_00486_RIGHT_MLO', 'P_00666_LEFT_CC', 'P_01683_LEFT_CC', 'P_00020_LEFT_MLO', 'P_01029_RIGHT_MLO', 'P_01823_RIGHT_CC',
         'P_01307_LEFT_MLO', 'P_00011_LEFT_MLO', 'P_00522_RIGHT_MLO', 'P_00557_LEFT_CC', 'P_01406_RIGHT_CC', 'P_01810_LEFT_CC',
         'P_00224_LEFT_MLO', 'P_01596_LEFT_CC', 'P_01526_RIGHT_CC', 'P_01818_LEFT_CC', 'P_01563_RIGHT_MLO', 'P_02154_LEFT_CC',
         'P_01172_LEFT_MLO', 'P_00251_RIGHT_CC', 'P_01346_LEFT_MLO', 'P_00877_LEFT_CC', 'P_02489_RIGHT_CC', 'P_00606_LEFT_MLO',
         'P_00357_LEFT_CC', 'P_00624_RIGHT_MLO', 'P_00400_RIGHT_MLO', 'P_00669_RIGHT_CC', 'P_00085_RIGHT_CC', 'P_00609_RIGHT_MLO',
         'P_00049_RIGHT_CC', 'P_00895_RIGHT_CC', 'P_01043_RIGHT_CC', 'P_00685_RIGHT_CC', 'P_00008_RIGHT_MLO', 'P_01629_LEFT_MLO',
         'P_00832_RIGHT_MLO', 'P_01488_LEFT_MLO', 'P_01440_LEFT_CC', 'P_00471_LEFT_MLO', 'P_02405_RIGHT_MLO', 'P_00084_LEFT_MLO',
         'P_00484_LEFT_MLO', 'P_00937_LEFT_MLO', 'P_01639_RIGHT_MLO', 'P_01116_LEFT_CC', 'P_00455_LEFT_CC', 'P_01596_LEFT_MLO',
         'P_00020_LEFT_CC', 'P_01274_LEFT_MLO', 'P_01709_RIGHT_CC', 'P_00571_RIGHT_CC', 'P_01457_RIGHT_CC', 'P_00185_LEFT_CC',
         'P_01313_LEFT_CC', 'P_01777_LEFT_MLO', 'P_02348_RIGHT_CC', 'P_01575_RIGHT_MLO', 'P_00062_LEFT_CC', 'P_01186_LEFT_CC',
         'P_00220_LEFT_MLO', 'P_00855_RIGHT_CC', 'P_00029_LEFT_CC', 'P_02154_LEFT_MLO', 'P_00030_LEFT_MLO', 'P_00196_RIGHT_CC',
         'P_00861_RIGHT_CC', 'P_00546_RIGHT_MLO', 'P_01095_LEFT_CC', 'P_00208_RIGHT_MLO', 'P_00583_LEFT_CC', 'P_00960_LEFT_CC',
         'P_00835_LEFT_CC', 'P_02218_LEFT_CC', 'P_01664_LEFT_CC', 'P_00162_LEFT_CC', 'P_02402_RIGHT_CC', 'P_00316_RIGHT_CC',
         'P_01809_LEFT_MLO', 'P_01769_LEFT_MLO', 'P_00736_LEFT_MLO', 'P_00652_LEFT_MLO', 'P_02234_LEFT_MLO', 'P_00825_RIGHT_MLO',
         'P_00098_RIGHT_CC', 'P_00016_LEFT_CC', 'P_00438_LEFT_CC', 'P_02313_LEFT_MLO', 'P_01792_RIGHT_MLO', 'P_00944_LEFT_MLO',
         'P_00655_RIGHT_CC', 'P_02259_LEFT_MLO', 'P_01576_RIGHT_MLO', 'P_00945_RIGHT_CC', 'P_01664_RIGHT_MLO', 'P_01297_LEFT_MLO',
         'P_01458_LEFT_MLO', 'P_00256_LEFT_MLO', 'P_00573_RIGHT_MLO', 'P_01467_RIGHT_MLO', 'P_01422_RIGHT_CC', 'P_01767_LEFT_CC',
         'P_00563_RIGHT_MLO', 'P_01256_RIGHT_MLO', 'P_00878_LEFT_CC', 'P_01628_LEFT_CC', 'P_00088_LEFT_MLO', 'P_01128_RIGHT_CC',
         'P_00527_RIGHT_MLO', 'P_00083_RIGHT_CC', 'P_00978_LEFT_MLO', 'P_01029_RIGHT_CC', 'P_01274_RIGHT_CC', 'P_00792_LEFT_CC',
         'P_01401_LEFT_CC', 'P_00123_LEFT_MLO', 'P_01137_LEFT_CC', 'P_01186_LEFT_MLO', 'P_00316_RIGHT_MLO', 'P_01049_LEFT_CC',
         'P_00019_RIGHT_MLO', 'P_01345_LEFT_CC', 'P_00768_LEFT_CC', 'P_01537_RIGHT_CC', 'P_01337_LEFT_CC', 'P_01691_RIGHT_CC',
         'P_00106_RIGHT_MLO', 'P_02226_LEFT_CC', 'P_00864_LEFT_CC', 'P_00098_LEFT_MLO', 'P_01053_LEFT_MLO', 'P_00897_LEFT_MLO',
         'P_02218_LEFT_MLO', 'P_01839_LEFT_MLO', 'P_00924_LEFT_CC'
         ]


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
    STEPS_PER_EPOCH = (1229 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

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

    def load_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)


    def load_mammo(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # # Add classes. We have one class.
        # # Naming the dataset mass, and the class mass
        # self.add_class("mass", 1, "mass")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "mass_train", "mass_test", "calc_test", "stage2_test"]
        # subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        subset_dir = "mass_train" if subset in ["mass_train", "val"] else subset
        # subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        print(subset)
        image_ids = next(os.walk(dataset_dir))[1]
        val_size = round(len(image_ids)* 0.20)
        np.random.seed(0)
        np.random.shuffle(image_ids)
        VAL_IMAGE_IDS = image_ids[:val_size]

        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            # image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train" or subset == "mass_train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add classes. We have one class.
        # Naming the dataset mass, and the class mass
        self.add_class("mass", 1, "mass")

        # Add images
        for image_id in image_ids:
            self.add_image(
                "mass",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "full_image/{}_full.png".format(image_id)))
                # path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                if(m.ndim == 2):
                    mask.append(m)
                else: # We only want 2 dimension image since the mask is grayscale
                    mask.append(m[:,:,0])

        # print("Mask shape before np.stack:", mask)
        mask = np.stack(mask, axis=-1)
        # print("Mask shape after np.stack:", mask.shape)

        # print(mask.shape)
        # If mask is 4-dimension, make it 3 dimension only
        if mask.ndim == 4:
            mask = mask[:,:,:,0]

        # print("Mask shape after reshaping:", mask.shape)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

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
    dataset_val.load_mammo(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((2, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
        ,iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        # iaa.Multiply((0.8, 1.5)),
        # iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
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
