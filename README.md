# Breast Cancer Segmentation with Mask R-CNN

This is a project for my thesis dissertation for the MSc program in Data Science and Machine Learning at University College London.

This project uses the [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) architecture to investigate if the model is able to perform instance segmentation on mass lesions in mammogram images. The Mask R-CNN model has been optimized for the natural
image domain and we are interested to know if it transposes to the domain of mammograms.

## Supervisors
* Stephen Morrell
* Dr. Juan Eugenio

## Dependencies
* Python 3
* TensorFlow
* Keras

## Data set
* [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)

## Folder structure
```
|--mrcnn
|--faster_rcnn
|--mammography
|--dataset (excluded in repo)
    |--mammo
        |--mass_train
        |--mass_test
            | P_XXXXX_LEFT_CC
                |--full_image
                    |--P_XXXXX_LEFT_CC.png
                |--masks
                    |--P_XXXXX_LEFT_CC_1.png
            | ...
```
