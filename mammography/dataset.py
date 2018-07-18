import sys
sys.path.append('../')
import utils
import numpy as np
import imageio
import os
import skimage

class MammoDataset(utils.Dataset):
    """Override:
            load_image()
            load_mask()
            image_reference()
    """

    def add_mass(self, root_dir, mode, split_ratio=0.9):
        # Add classes
        self.add_class("mass", 1, "mass") # source, id, name. id = 0s is BG

        image_names = os.listdir(root_dir)
        length = len(image_names)

        np.random.seed(1000)
        image_names = list(np.random.permutation(image_names))
        np.random.seed(None)

        if mode == 'train':
            image_names = image_names[: int(split_ratio*length)]
        if mode == 'val':
            image_names = image_names[int(split_ratio*length):]
        if mode == 'val_as_test':
            image_names = image_names[int(split_ratio*length):]
            mode = 'test'
        dirs = [root_dir + img_name + '/full_image/' for img_name in image_names]
        mask_dirs = [root_dir + img_name + '/masks/' for img_name in image_names]

        # Add images
        for i in range(len(image_names)):
            self.add_image(
                source = "mass",
                image_id = i,
                path = dirs[i] + image_names[i] + '.png',
                mask_dir = mask_dirs[i],
                name = image_names[i]
                )


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = imageio.imread(self.image_info[image_id]['path'])
        # RGBA to RGB
        if image.shape[2] != 3:
            image = image[:,:,:3]
        image = self.preprocess(image)
        return image

    def image_reference(self, image_id):
        """Return the details of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mass":
            return info["path"]
        else:
            super(MammoDataset, self).image_reference(self, image_id)

    def load_mask(self, image_id):
        """
        Returns:
            masks: A binary array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_dir= info['mask_dir']
        mask_names = os.listdir(mask_dir)
        mask_paths = [mask_dir + mask_name for mask_name in mask_names]

        count = len(mask_paths)

        # masks = [imageio.imread(path) for path in mask_paths]
        masks = []
        for path in mask_paths:
            msk = imageio.imread(path)
            if np.sum(msk) == 0:
                continue
            msk = msk.astype('float32')/255.
            masks.append(msk)
        # mask = np.stack(masks, axis=-1)
#        mask = mask.astype(bool)
        # mask = np.where(mask>128, 1, 0)
        masks = np.asarray(masks)
        masks[masks > 0.] = 1.
        masks = np.transpose(masks, (1,2,0))
        occlusion = np.logical_not(masks[:, : ,-1]).astype(np.uint8)
        count = masks.shape[2]
        for i in range(count-2, -1, -1):
            masks[:, :, i] = masks[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
        class_ids = [self.class_names.index('mass') for s in range(count)]
        class_ids = np.asarray(class_ids)

        # class_ids = np.ones(count,dtype=np.int32)
        return masks, class_ids

    def preprocess(self, img):
        gray = skimage.color.rgb2gray(img.astype('uint8'))
        img = skimage.color.gray2rgb(gray)
        img *= 255.
        return img

    def load_semantic(self, image_id):
        info = self.image_info[image_id]
        path = info['mask_dir'].replace('masks','full_image')
        mask_path = path + 'mask.png'


        mask = imageio.imread(mask_path)
        mask = np.where(mask>128, 1, 0)
        return mask

if __name__ == "__main__":
    ds = MammoDataset()
    #ds.add_nuclei('data/stage1_train/','train')
    ds.add_mass('data/mass_train/','train')
    ds.prepare()
    print(ds.image_info[0])

    image = ds.load_image(0)
    print(image.shape)

#    mask, _ = ds.load_mask(0)
#    print(len(_))
#    print(mask.shape)

    means = []
    for idx in ds.image_ids:
        im = ds.load_image(idx)
        means.append(np.mean(im[:,-1],axis=0))
    print(np.mean(means,axis=0))
