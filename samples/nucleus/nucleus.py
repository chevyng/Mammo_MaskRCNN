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

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/mammo/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = ['P_00893_LEFT_CC' ,'P_01273_RIGHT_CC' ,'P_00931_LEFT_CC' ,'P_01589_LEFT_CC'
 ,'P_01164_RIGHT_CC' ,'P_00601_LEFT_MLO' ,'P_00294_LEFT_MLO'
 ,'P_01142_RIGHT_CC' ,'P_00717_RIGHT_MLO' ,'P_01210_RIGHT_MLO'
 ,'P_01138_RIGHT_MLO' ,'P_01057_RIGHT_CC' ,'P_01035_RIGHT_CC'
 ,'P_00952_RIGHT_CC' ,'P_00194_RIGHT_MLO' ,'P_00059_LEFT_CC'
 ,'P_00287_RIGHT_CC' ,'P_01646_LEFT_MLO' ,'P_00317_RIGHT_MLO'
 ,'P_01395_RIGHT_CC' ,'P_01539_RIGHT_CC' ,'P_00437_LEFT_CC' ,'P_00711_LEFT_CC'
 ,'P_00515_LEFT_CC' ,'P_00773_LEFT_CC' ,'P_01323_LEFT_CC' ,'P_01356_LEFT_CC'
 ,'P_01222_LEFT_MLO' ,'P_01233_LEFT_MLO' ,'P_00218_LEFT_MLO'
 ,'P_00596_LEFT_CC' ,'P_01150_LEFT_CC' ,'P_01317_RIGHT_MLO'
 ,'P_00429_LEFT_MLO' ,'P_01790_LEFT_MLO' ,'P_01600_RIGHT_CC'
 ,'P_00510_LEFT_CC' ,'P_00099_LEFT_MLO' ,'P_00637_RIGHT_CC'
 ,'P_01512_LEFT_MLO' ,'P_01735_RIGHT_MLO' ,'P_00559_LEFT_CC'
 ,'P_00532_LEFT_CC' ,'P_00932_LEFT_MLO' ,'P_01008_RIGHT_MLO'
 ,'P_00636_RIGHT_MLO' ,'P_01187_LEFT_MLO' ,'P_00160_LEFT_MLO'
 ,'P_01634_LEFT_MLO' ,'P_01204_RIGHT_MLO' ,'P_00747_LEFT_MLO'
 ,'P_00110_LEFT_MLO' ,'P_01153_RIGHT_CC' ,'P_00487_RIGHT_CC'
 ,'P_01233_LEFT_CC' ,'P_01645_RIGHT_CC' ,'P_00369_LEFT_CC'
 ,'P_00435_RIGHT_MLO' ,'P_01071_LEFT_CC' ,'P_01383_LEFT_CC'
 ,'P_01623_RIGHT_CC' ,'P_00914_LEFT_CC' ,'P_01373_RIGHT_CC'
 ,'P_00190_LEFT_MLO' ,'P_00535_LEFT_CC' ,'P_01333_LEFT_CC' ,'P_01434_LEFT_MLO'
 ,'P_01829_LEFT_MLO' ,'P_01800_LEFT_MLO' ,'P_01477_LEFT_CC'
 ,'P_00678_LEFT_MLO' ,'P_00343_LEFT_MLO' ,'P_00775_LEFT_MLO'
 ,'P_01270_LEFT_CC' ,'P_00383_LEFT_MLO' ,'P_00287_RIGHT_MLO'
 ,'P_00634_LEFT_CC' ,'P_01509_RIGHT_MLO' ,'P_01394_RIGHT_CC'
 ,'P_01381_RIGHT_MLO' ,'P_01251_LEFT_MLO' ,'P_00700_RIGHT_MLO'
 ,'P_01427_LEFT_MLO' ,'P_01671_LEFT_MLO' ,'P_01138_RIGHT_CC'
 ,'P_00429_LEFT_CC' ,'P_00616_LEFT_MLO' ,'P_01130_RIGHT_CC' ,'P_01650_LEFT_CC'
 ,'P_01644_LEFT_MLO' ,'P_01749_LEFT_CC' ,'P_01213_LEFT_MLO'
 ,'P_01768_LEFT_MLO' ,'P_01426_RIGHT_CC' ,'P_01633_RIGHT_MLO'
 ,'P_00235_RIGHT_CC' ,'P_01179_LEFT_CC' ,'P_01267_RIGHT_MLO'
 ,'P_01052_LEFT_MLO' ,'P_01285_RIGHT_MLO' ,'P_00230_RIGHT_CC'
 ,'P_01650_LEFT_MLO' ,'P_00798_RIGHT_CC' ,'P_00090_LEFT_CC' ,'P_00451_LEFT_CC'
 ,'P_01833_RIGHT_MLO' ,'P_00313_RIGHT_MLO' ,'P_00116_RIGHT_CC'
 ,'P_01322_RIGHT_CC' ,'P_00543_RIGHT_CC' ,'P_00695_RIGHT_CC'
 ,'P_00661_LEFT_CC' ,'P_01330_RIGHT_MLO' ,'P_00409_RIGHT_CC'
 ,'P_00702_RIGHT_CC' ,'P_00450_LEFT_CC' ,'P_01504_LEFT_MLO' ,'P_00363_LEFT_CC'
 ,'P_01581_LEFT_CC'  ,'P_01553_RIGHT_MLO'
 ,'P_00348_LEFT_MLO' ,'P_00436_LEFT_MLO' ,'P_00901_RIGHT_MLO'
 ,'P_01326_LEFT_CC' ,'P_00592_LEFT_CC' ,'P_01194_LEFT_MLO' ,'P_00837_RIGHT_CC'
 ,'P_00340_LEFT_CC' ,'P_00332_RIGHT_CC' ,'P_01697_LEFT_CC'
 ,'P_00057_RIGHT_MLO' ,'P_00041_LEFT_CC' ,'P_00209_LEFT_MLO'
 ,'P_01051_RIGHT_MLO' ,'P_00913_LEFT_CC' ,'P_00283_RIGHT_MLO'
 ,'P_00801_LEFT_MLO' ,'P_00498_LEFT_MLO' ,'P_00389_LEFT_MLO'
 ,'P_01000_RIGHT_MLO' ,'P_01248_LEFT_CC' ,'P_00376_RIGHT_CC'
 ,'P_00241_RIGHT_CC' ,'P_01120_LEFT_MLO' ,'P_00528_RIGHT_CC'
 ,'P_00543_RIGHT_MLO' ,'P_00841_RIGHT_CC' ,'P_00366_RIGHT_MLO'
 ,'P_00607_RIGHT_CC' ,'P_01754_RIGHT_CC' ,'P_00409_RIGHT_MLO'
 ,'P_00718_RIGHT_MLO' ,'P_00032_RIGHT_CC' ,'P_01518_LEFT_CC'
 ,'P_01302_RIGHT_CC' ,'P_01327_LEFT_MLO' ,'P_00319_LEFT_CC'
 ,'P_00081_RIGHT_CC' ,'P_01162_LEFT_CC' ,'P_01225_RIGHT_MLO'
 ,'P_01761_LEFT_MLO' ,'P_01506_LEFT_MLO' ,'P_01343_LEFT_MLO'
 ,'P_00026_LEFT_MLO' ,'P_00177_LEFT_MLO' ,'P_02092_LEFT_MLO'
 ,'P_00598_LEFT_CC' ,'P_01656_LEFT_CC' ,'P_00623_LEFT_MLO' ,'P_01778_RIGHT_CC'
 ,'P_00851_LEFT_MLO' ,'P_00820_LEFT_MLO' ,'P_01715_RIGHT_CC'
 ,'P_01238_RIGHT_CC' ,'P_00396_LEFT_CC' ,'P_00572_RIGHT_CC'
 ,'P_00694_RIGHT_CC' ,'P_00942_LEFT_CC' ,'P_01532_LEFT_CC'
 ,'P_01114_RIGHT_MLO' ,'P_01325_RIGHT_MLO' ,'P_00987_LEFT_MLO'
 ,'P_00454_RIGHT_MLO' ,'P_01590_LEFT_MLO' ,'P_01591_LEFT_MLO'
 ,'P_01297_LEFT_MLO' ,'P_01144_LEFT_CC' ,'P_00499_RIGHT_CC'
 ,'P_00120_LEFT_MLO' ,'P_01780_LEFT_CC' ,'P_01878_RIGHT_CC'
 ,'P_00303_RIGHT_MLO' ,'P_01165_LEFT_MLO' ,'P_00086_RIGHT_CC'
 ,'P_01265_RIGHT_MLO' ,'P_01819_RIGHT_CC' ,'P_01788_RIGHT_MLO'
 ,'P_00920_RIGHT_MLO' ,'P_01485_LEFT_MLO' ,'P_00889_LEFT_MLO'
 ,'P_01207_RIGHT_MLO' ,'P_00401_LEFT_MLO' ,'P_01406_LEFT_MLO'
 ,'P_01540_LEFT_MLO' ,'P_00095_LEFT_MLO' ,'P_01268_RIGHT_MLO'
 ,'P_01877_LEFT_CC' ,'P_00823_RIGHT_CC' ,'P_00430_LEFT_CC'
 ,'P_00328_RIGHT_MLO' ,'P_01534_RIGHT_CC' ,'P_01290_RIGHT_CC'
 ,'P_00863_RIGHT_CC' ,'P_00922_RIGHT_CC' ,'P_00644_LEFT_CC'
 ,'P_01411_RIGHT_MLO' ,'P_00826_LEFT_MLO' ,'P_01052_LEFT_CC'
 ,'P_00553_LEFT_MLO' ,'P_00061_RIGHT_CC' ,'P_01581_LEFT_MLO'
 ,'P_01981_RIGHT_MLO' ,'P_00359_LEFT_MLO' ,'P_01150_LEFT_MLO'
 ,'P_00406_RIGHT_MLO' ,'P_00408_RIGHT_CC' ,'P_00707_RIGHT_CC'
 ,'P_01807_RIGHT_MLO' ,'P_01378_RIGHT_MLO' ,'P_00951_RIGHT_MLO'
 ,'P_01386_LEFT_CC' ,'P_00045_LEFT_MLO' ,'P_01364_LEFT_MLO'
 ,'P_00224_RIGHT_MLO' ,'P_01517_LEFT_MLO'
 ,'P_01083_RIGHT_MLO']
# VAL_IMAGE_IDS = [
#     "P_00016_LEFT_MLO",
#     "P_00017_LEFT_CC",
#     "P_00147_RIGHT_CC",
#     "P_00156_RIGHT_MLO",
#     "P_00173_LEFT_CC",
#     "P_00200_RIGHT_CC",
#     "P_00209_LEFT_MLO",
#     "P_00230_RIGHT_MLO",
#     "P_00544_LEFT_CC",
#     "P_00601_LEFT_MLO",
#     "P_00738_RIGHT_CC",
#     "P_00837_RIGHT_MLO",
#     "P_00198_LEFT_CC",
#     "P_00296_LEFT_CC",
#     "P_00296_LEFT_MLO",
#     "P_00358_RIGHT_MLO",
#     "P_00359_LEFT_MLO",
#     "P_00387_RIGHT_MLO",
#     "P_00470_RIGHT_CC",
#     "P_00482_LEFT_MLO",
#     "P_00516_LEFT_MLO",
#     "P_00524_LEFT_MLO",
#     "P_00524_LEFT_CC",
#     "P_00612_RIGHT_MLO",
#     'P_00615_RIGHT_CC',
#     "P_00623_LEFT_CC",
#     "P_00677_RIGHT_MLO",
#     "P_00699_RIGHT_CC",
#     "P_00718_RIGHT_MLO",
# ]


# VAL_IMAGE_IDS = [
#     "0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2",
#     "92f31f591929a30e4309ab75185c96ff4314ce0a7ead2ed2c2171897ad1da0c7",
#     "1e488c42eb1a54a3e8412b1f12cde530f950f238d71078f2ede6a85a02168e1f",
#     "c901794d1a421d52e5734500c0a2a8ca84651fb93b19cec2f411855e70cae339",
#     "8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b",
#     "60cb718759bff13f81c4055a7679e81326f78b6a193a2d856546097c949b20ff",
#     "da5f98f2b8a64eee735a398de48ed42cd31bf17a6063db46a9e0783ac13cd844",
#     "9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32",
#     "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",
#     "97126a9791f0c1176e4563ad679a301dac27c59011f579e808bbd6e9f4cd1034",
#     "e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0",
#     "f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81",
#     "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1",
#     "3ab9cab6212fabd723a2c5a1949c2ded19980398b56e6080978e796f45cbbc90",
#     "ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716",
#     "bb61fc17daf8bdd4e16fdcf50137a8d7762bec486ede9249d92e511fcb693676",
#     "e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b",
#     "947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050",
#     "cbca32daaae36a872a11da4eaff65d1068ff3f154eedc9d3fc0c214a4e5d32bd",
#     "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",
#     "4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06",
#     "f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af",
#     "a4c44fc5f5bf213e2be6091ccaed49d8bf039d78f6fbd9c4d7b7428cfcb2eda4",
#     "cab4875269f44a701c5e58190a1d2f6fcb577ea79d842522dcab20ccb39b7ad2",
#     "8ecdb93582b2d5270457b36651b62776256ade3aaa2d7432ae65c14f07432d49",
# ]



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
    STEPS_PER_EPOCH = (1584 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
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
    RPN_NMS_THRESHOLD = 0.5


############################################################
#  Dataset
############################################################

class MammoDataset(utils.Dataset):

    def load_mammo(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset mass, and the class mass
        self.add_class("mass", 1, "mass")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        print(subset)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            # np.random.shuffle(image_ids)
            # image_ids = np.array(image_ids)
            # VAL_IMAGE_IDS = image_ids[0:int(len(image_ids)*0.15)]
            # print(VAL_IMAGE_IDS)
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

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
                mask.append(m)

        mask = np.stack(mask, axis=-1)
        # print(mask.shape)
        # If mask is 4-dimension, make it 3 dimension only
        if mask.ndim == 4:
            mask = mask[:,:,:,0]
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
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
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
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MammoDataset()
    dataset.load_mammo(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


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
