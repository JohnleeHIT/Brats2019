from __future__ import division
import numpy as np
import nibabel as nib
import copy
import time
import configparser
from skimage.transform import resize
from scipy.ndimage import measurements
import tensorflow as tf
from glob import glob
import re
import SimpleITK as sitk
import random
from keras_preprocessing.image import *
import cv2 as cv
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import Counter


# construct a iterator for batch generation
class BatchGenerator(Iterator):
    '''
    get an iteratator for generating（batch_x, batch_y）
    '''

    def __init__(
            self,
            batch_size,
            shuffle,
            seed,
            volume_path,
            modalities,
            resize_r,
            rename_map,
            patch_dim,
            augmentation):
        self.batch_size = batch_size
        self.volume_path = volume_path
        self.modalities = modalities
        self.resize_ratio = resize_r
        self.rename_map = rename_map
        self.file_list = self._get_img_info()
        self.total_num = len(self.file_list)
        self.patch_dim = patch_dim
        # self.rot_flag = rot_flag
        self.augmentation = augmentation
        self.image_shape = (patch_dim, patch_dim, patch_dim) + (modalities,)
        self.label_shape = (patch_dim, patch_dim, patch_dim)
        super(
            BatchGenerator,
            self).__init__(
            n=self.total_num,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)

    def _get_img_info(self):
        '''
        this function read all files of specific directory, get the path list
        :return:path list of all the volume files
        '''
        file_list = []
        categories = os.listdir(self.volume_path)
        for category in categories:
            category_path = os.path.join(self.volume_path, category)
            dir_list = os.listdir(category_path)
            for dire in dir_list:
                dire_lower = dire.lower()
                if not dire_lower.startswith('brats'):
                    raise Exception("volume file exception!")
                file_abs_path = os.path.join(category_path, dire)
                single_file = {"path": file_abs_path, "category": category}
                file_list.append(single_file)

        return file_list

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros(
            (len(index_array),
             ) + self.image_shape,
            dtype='float32')
        batch_x2 = np.zeros(
            (len(index_array),
             ) + self.label_shape+(1,),
            dtype='float32')
        batch_y = np.zeros(
            (len(index_array),
             ) + self.label_shape,
            dtype='int32')
        batch_y_stage2 = np.zeros(
            (len(index_array),
             ) + self.label_shape,
            dtype='int32')
        batch_y_stage3 = np.zeros(
            (len(index_array),
             ) + self.label_shape,
            dtype='int32')
        for i, j in enumerate(index_array):
            # data directory of a patient
            single_dir_path = self.file_list[j]["path"]

            img_data, img_data2, stage1_label_data, stage2_label, \
            stage3_label, _ = self.load_volumes_label(single_dir_path, True)

            rand_num = np.random.randint(self.total_num - 1, size=self.total_num)
            matching_index = rand_num[0] if rand_num[0] != j else rand_num[-1]
            # ready for histogram matching
            img_data_matching, img_data_matching2, _, _, _, _ = self.load_volumes_label(
                self.file_list[matching_index]["path"], True)
            img_data_matching_cast = img_data_matching.astype("float32")
            img_data_matching_cast2 = img_data_matching2.astype("float32")

            # data augmentation
            volume_list = [img_data[...,0], img_data[...,1], np.squeeze(img_data2, axis=-1),
                           stage1_label_data, stage2_label, stage3_label]
            img_data_0, img_data_1, img_data2, stage1_label_data, \
            stage2_label, stage3_label = self.data_augment_volume(*volume_list,
                                                                 augmentation=self.augmentation)

            img_data = np.stack((img_data_0,img_data_1), axis=-1)
            img_data2 = np.expand_dims(img_data2, axis=-1)

            # reduce background region
            regions = get_brain_region(np.squeeze(img_data2, axis=-1))
            img_data = img_data[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            img_data2 = img_data2[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            stage1_label_data = stage1_label_data[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]
            stage2_label= stage2_label[regions[0]:regions[1], regions[2]:regions[3],
                                   regions[4]:regions[5]]
            stage3_label = stage3_label[regions[0]:regions[1], regions[2]:regions[3],
                               regions[4]:regions[5]]

            # test whether using the histogram matching data augmentation method.(deprecated)
            augment = False
            if augment:
                # histogram matching data augmentation
                img_hist_match = Preprocessing.hist_match(
                    img_data.astype("float32"), img_data_matching_cast)

                img_hist_match2 = Preprocessing.hist_match(img_data2.astype("float32"), img_data_matching_cast2)
                # using B-spine interpolation for deformation (just like V-net did)
                numcontrolpoints = 2
                sigma = 15
            else:
                img_hist_match = img_data
                img_hist_match2 = img_data2

            # resize
            resize_dim = (np.array(stage1_label_data.shape) * self.resize_ratio).astype('int')
            img_data_resize = resize(img_hist_match.astype("float32"), resize_dim, order=1, preserve_range=True)
            img_data2_resize = resize(img_hist_match2.astype("float32"), resize_dim, order=1, preserve_range=True)

            stage1_label_resize = resize(stage1_label_data, resize_dim, order=0, preserve_range=True)
            stage2_label_resize = resize(stage2_label, resize_dim, order=0, preserve_range=True)
            stage3_label_resize = resize(stage3_label, resize_dim, order=0, preserve_range=True)

            img_data_cast = img_data_resize.astype("float32")
            img_data_cast2 = img_data2_resize.astype("float32")

            label_data_cast = stage1_label_resize.astype('int32')
            stage2_label_cast = stage2_label_resize.astype("int32")
            stage3_label_cast = stage3_label_resize.astype("int32")

            # normalization
            img_norm = Preprocessing.Normalization(img_data_cast, axis=(0, 1, 2))
            img_norm2 = Preprocessing.Normalization(img_data_cast2)
            # randomly select a box anchor
            l, w, h = label_data_cast.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)

            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(img_norm[pos[0]:pos[0] +
                                                   self.patch_dim, pos[1]:pos[1] +
                                                   self.patch_dim, pos[2]:pos[2] +
                                                   self.patch_dim, :])
            img_temp2 = copy.deepcopy(img_norm2[pos[0]:pos[0] +
                                                   self.patch_dim, pos[1]:pos[1] +
                                                   self.patch_dim, pos[2]:pos[2] +
                                                   self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(
                label_data_cast[pos[0]:pos[0] + self.patch_dim, pos[1]:pos[1] + self.patch_dim, pos[2]:pos[2] + self.patch_dim])

            stage2_label_temp = copy.deepcopy(stage2_label_cast[pos[0]:pos[0] + self.patch_dim, pos[1]:pos[1] + self.patch_dim, pos[2]:pos[2] + self.patch_dim])
            stage3_label_temp = copy.deepcopy(stage3_label_cast[pos[0]:pos[0] + self.patch_dim, pos[1]:pos[1] + self.patch_dim, pos[2]:pos[2] + self.patch_dim])

            # get the batch data
            batch_x[i, :, :, :, :] = img_temp
            batch_x2[i, :, :, :, :] = img_temp2
            batch_y[i, :, :, :] = label_temp
            batch_y_stage2[i,:,:,:] = stage2_label_temp
            batch_y_stage3[i,:,:,:] = stage3_label_temp

        return batch_x, batch_x2, batch_y, batch_y_stage2, batch_y_stage3

    # load volumes and the GT
    def load_volumes_label(self, src_path, rename_map_flag):
        '''
        this function get the volume data and gt from the giving path
        :param src_path: directory path of a patient
        :return: GT and the volume data（width,height, slice, modality）
        '''
        # rename_map = [0, 1, 2, 4]
        volume_list, seg_dict = self.data_dict_construct(src_path)
        # assert len(volume_list) == 4
        # assert seg_dict["mod"] == "seg"
        if seg_dict["mod"] == "seg":
            label_nib_data = nib.load(seg_dict["path"])
            label = label_nib_data.get_data().copy()
            # label = nib.load(seg_dict["path"]).get_data().copy()

            # resolve the issue from resizing label, we first undertake binarization and then resize
            stage1_label_data = np.zeros(label.shape, dtype='int32')
            stage2_label_data = np.zeros(label.shape, dtype='int32')
            stage3_label_data = np.zeros(label.shape, dtype='int32')
            if rename_map_flag:
                for i in range(len(self.rename_map)):
                    if i > 0:
                        stage1_label_data[label == self.rename_map[i]] = 1
                    else:
                        continue
                # Cascaded structure，stage2,stage3 label prepare
                stage2_label_data[label == 1] = 1
                stage2_label_data[label == 4] = 1
                stage3_label_data[label == 1] = 1
            else:
                stage1_label_data = copy.deepcopy(label).astype('int16')
                stage2_label_data = copy.deepcopy(label).astype('int16')
                stage3_label_data = copy.deepcopy(label).astype('int16')
        else:
            stage1_label_data = []
            stage2_label_data = []
            stage3_label_data = []
            label_nib_data = []

        img_all_modality = []
        # order of the sequences [flair, T1, T1ce, T2]
        for i in range(len(volume_list)):
            volume = nib.load(volume_list[i]["path"])
            img = volume.get_data().copy()
            # resized_img = resize(img, resize_dim, order=1, preserve_range=True)
            img_all_modality.append(img)

        # choose different modalities for the network
        if self.modalities == 4:
            # all the modalities
            img_data = img_all_modality
        elif self.modalities == 3:
            # select T1ce T1 Flair modalities
            img_data = [img_all_modality[0], img_all_modality[2], img_all_modality[3]]
        elif self.modalities == 2:
            # two modalities
            # choose T2 and Flair
            img_data = [img_all_modality[0], img_all_modality[3]]
        else:
            # one modality
            img_data = img_all_modality[0]
            img_data = np.expand_dims(img_data, axis=0)

        # input volume data
        img_data2 = np.expand_dims(img_all_modality[2], axis=0)
        img_array2 = np.array(img_data2, "float32").transpose((1,2,3,0))
        # list to ndarray
        img_array = np.array(img_data, "float32").transpose((1, 2, 3, 0))
        return img_array, img_array2, stage1_label_data, stage2_label_data, stage3_label_data, volume

    # construct data dict
    def data_dict_construct(self, path):
        '''
        this function get the list of dictionary of the patients
        :param path: path of the patient data
        :return: list of dictionary including the path and the modality
        '''
        # list the image volumes and GT
        files = os.listdir(path)
        nii_list = sorted(glob('{}/*.nii.gz'.format(path)))
        re_style = r'[\-\_\.]+'
        volumn_list = []
        seg_dict = {"mod": "None"}
        for count, nii in enumerate(nii_list):
            # modality mapping [seg, flair, T1, T1ce, T2]
            mapping = [0, 1, 2, 3, 4]
            file = os.path.basename(nii)
            split_text = re.split(re_style, file)
            modality = split_text[-3]
            assert modality in ["flair", "seg", "t1", "t2", "t1ce"]
            if modality == "seg":
                data_dict = {"mod": modality, "path": nii, "count": mapping[0]}
            elif modality == "flair":
                data_dict = {"mod": modality, "path": nii, "count": mapping[1]}
            elif modality == "t1":
                data_dict = {"mod": modality, "path": nii, "count": mapping[2]}
            elif modality == "t1ce":
                data_dict = {"mod": modality, "path": nii, "count": mapping[3]}
            else:
                data_dict = {"mod": modality, "path": nii, "count": mapping[4]}

            if data_dict["mod"] != "seg":
                volumn_list.append(data_dict)
            else:
                seg_dict = {"mod": modality, "path": nii, "count": mapping[0]}
        # sort the modalites in the list
        volumn_list.sort(key=lambda x: x["count"])
        return volumn_list, seg_dict


    def data_augment_volume(self, *datalist , augmentation):

        # first get the volume data from the data list
        image1, image2, image3, mask1, mask2, mask3 = datalist
        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image1_shape = image1.shape
            mask1_shape = mask1.shape
            image2_shape = image2.shape
            mask2_shape = mask2.shape
            image3_shape = image3.shape
            mask3_shape = mask3.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            # image should be uint8!!
            image1 = det.augment_image(image1)
            image2 = det.augment_image(image2)
            image3 = det.augment_image(image3)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask1 = det.augment_image(mask1.astype(np.uint8),
                                       hooks=imgaug.HooksImages(activator=hook))
            mask2 = det.augment_image(mask2.astype(np.uint8),
                                      hooks=imgaug.HooksImages(activator=hook))
            mask3 = det.augment_image(mask3.astype(np.uint8),
                                      hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image1.shape == image1_shape, "Augmentation shouldn't change image size"
            assert mask1.shape == mask1_shape, "Augmentation shouldn't change mask size"
            assert image2.shape == image2_shape, "Augmentation shouldn't change image size"
            assert mask2.shape == mask2_shape, "Augmentation shouldn't change mask size"
            assert image3.shape == image3_shape, "Augmentation shouldn't change image size"
            assert mask3.shape == mask3_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            # masks = masks.astype(np.bool)
        return image1,image2, image3, mask1, mask2, mask3

    def data_augment(self, image, mask, augmentation):
        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            # image should be uint8!!
            images = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            masks = det.augment_image(mask.astype(np.uint8),
                                       hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert images.shape == image_shape, "Augmentation shouldn't change image size"
            assert masks.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            # masks = masks.astype(np.bool)
        return image, mask


def get_brain_region(volume_data):
    # volume = nib.load(volume_path)
    # volume_data = volume.get_data()
    # get the brain region
    indice_list = np.where(volume_data > 0)
    # calculate the min and max of the indice,  here volume have 3 channels
    channel_0_min = min(indice_list[0])
    channel_0_max = max(indice_list[0])

    channel_1_min = min(indice_list[1])
    channel_1_max = max(indice_list[1])

    channel_2_min = min(indice_list[2])
    channel_2_max = max(indice_list[2])

    brain_volume = volume_data[channel_0_min:channel_0_max, channel_1_min:channel_1_max,channel_2_min:channel_2_max]

    return (channel_0_min, channel_0_max, channel_1_min, channel_1_max, channel_2_min, channel_2_max)


class Preprocessing(object):

    def __init__(self):
        pass

    # N4 Bias Field Correction by simpleITK
    @staticmethod
    def N4BiasFieldCorrection(src_path, dst_path):
        '''
        This function carry out BiasFieldCorrection for the files in a specific directory
        :param src_path: path of the source file
        :param dst_path: path of the target file
        :return:
        '''
        print("N4 bias correction runs.")
        inputImage = sitk.ReadImage(src_path)

        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        sitk.WriteImage(maskImage, dst_path)

        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        # corrector.SetMaximumNumberOfIterations(10)

        output = corrector.Execute(inputImage, maskImage)
        sitk.WriteImage(output, dst_path)
        print("Finished N4 Bias Field Correction.....")

    # normalize the data(zero mean and unit variance)
    @staticmethod
    def Normalization(volume, axis=None):
        mean = np.mean(volume, axis=axis)
        std = np.std(volume, axis=axis)
        norm_volume = (volume - mean) / std
        return norm_volume

    # data augmentation by histogram matching
    @staticmethod
    def hist_match(source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source（randomly choose from the training dataset）
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        # interp_t_values = np.zeros_like(source,dtype=float)
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    # data augmentation by deforming
    @staticmethod
    def produceRandomlyDeformedImage(image, label, numcontrolpoints, stdDef, seed=1):
        '''
        This function comes from V-net，deform a image by B-spine interpolation
        :param image: images ，numpy array
        :param label: labels，numpy array
        :param numcontrolpoints: control point，B-spine interpolation parameters，take 2 for default
        :param stdDef: Deviation，B-spine interpolation parameters，take 15 for default
        :return: Deformed images and GT in numpy array
        '''
        sitkImage = sitk.GetImageFromArray(image, isVector=False)
        sitklabel = sitk.GetImageFromArray(label, isVector=False)

        transfromDomainMeshSize = [numcontrolpoints] * sitkImage.GetDimension()

        tx = sitk.BSplineTransformInitializer(
            sitkImage, transfromDomainMeshSize)

        params = tx.GetParameters()

        paramsNp = np.asarray(params, dtype=float)
        # 设置种子值，确保多通道时两个通道变换程度一样
        np.random.seed(seed)
        paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * stdDef

        # remove z deformations! The resolution in z is too bad
        paramsNp[0:int(len(params) / 3)] = 0

        params = tuple(paramsNp)
        tx.SetParameters(params)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)

        resampler.SetDefaultPixelValue(0)
        outimgsitk = resampler.Execute(sitkImage)
        outlabsitk = resampler.Execute(sitklabel)

        outimg = sitk.GetArrayFromImage(outimgsitk)
        outimg = outimg.astype(dtype=np.float32)

        outlbl = sitk.GetArrayFromImage(outlabsitk)
        # outlbl = (outlbl > 0.5).astype(dtype=np.float32)

        return outimg, outlbl


class Evaluation(object):
    def __init__(self):
        pass

    # save 3d volume as slices
    def save_slice_img(self, volume_path, output_path):
        file_name = os.path.basename(volume_path)
        output_dir  = os.path.join(output_path, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            pass
        input_volume = nib.load(volume_path).get_data()
        # mapping to 0-1
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        input_unit = (input_volume-vol_min)/(vol_max - vol_min)
        width, height, depth= input_unit.shape
        for i in range(0, depth):
            slice_path = os.path.join(output_dir, str(i)+'.png')
            img_i = input_unit[:, :, i]
            # normalize to 0-255
            img_i = (img_i*255).astype('uint8')
            # cv.imwrite(slice_path, img_i)
        return input_unit

    def save_slice_img_label(self, img_volume, pre_volume, gt_volume,
                             output_path, file_name, show_mask=False, show_gt = False):
        assert img_volume.shape == pre_volume.shape
        if show_gt:
            assert img_volume.shape == gt_volume.shape
        width, height, depth = img_volume.shape
        # gray value mapping   from MRI value to pixel value(0-255)
        volume_max = np.max(img_volume)
        volume_min = np.min(img_volume)
        volum_mapped = (img_volume-volume_min)/(volume_max-volume_min)
        volum_mapped = (255*volum_mapped).astype('uint8')
        # construct a directory for each volume to save slices
        dir_volume = os.path.join(output_path, file_name)
        if not os.path.exists(dir_volume):
            os.makedirs(dir_volume)
        else:
            pass
        for i in range(depth):
            img_slice = volum_mapped[:, :, i]
            pre_slice = pre_volume[:, :, i]
            if show_gt:
                gt_slice = gt_volume[:, :, i]
            else:
                gt_slice = []
            self.save_contour_label(img=img_slice, pre=pre_slice, gt=gt_slice,
                                    save_path=dir_volume, file_name=i,show_mask=show_mask,show_gt=show_gt)

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def save_contour_label(self, img, pre, gt=None, save_path='', file_name=None, show_mask=False, show_gt = False):
        # single channel to multi-channel
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))

        # Generate random colors
        # colors = self.random_colors(4)
        # Prediction result is illustrated as red and the groundtruth is illustrated as blue
        colors = [[1.0, 0, 0], [0, 0, 1.0]]
        # Show area outside image boundaries.

        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')
        # ax.set_title("volume mask")
        masked_image = img.astype(np.uint32).copy()

        if show_mask:
            masked_image = self.apply_mask(masked_image, pre, colors[0])
            if show_gt:
                masked_image = self.apply_mask(masked_image, gt, colors[1])

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask_pre = np.zeros(
            (pre.shape[0] + 2, pre.shape[1] + 2), dtype=np.uint8)
        padded_mask_pre[1:-1, 1:-1] = pre
        contours = find_contours(padded_mask_pre, 0.5)
        for verts in contours:
            # reduce padding and  flipping from (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=colors[0], linewidth=1)
            ax.add_patch(p)

        if show_gt:
            padded_mask_gt = np.zeros((gt.shape[0] + 2, gt.shape[1] + 2), dtype=np.uint8)
            padded_mask_gt[1:-1, 1:-1] = gt
            contours_gt = find_contours(padded_mask_gt, 0.5)

            for contour in contours_gt:
                contour = np.fliplr(contour) -1
                p_gt = Polygon(contour, facecolor="none", edgecolor=colors[1], linewidth=1)
                ax.add_patch(p_gt)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        ax.imshow(masked_image.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, file_name))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def save_slice_volume(volume, save_path):
    '''
    the function save volume data to slices in the specific directory
    :param volume: input volume data
    :param save_path:
    :return:
    '''
    shape = volume.shape
    # translate intensity to 0-255
    v_max = np.max(volume)
    v_min = np.min(volume)
    volume_norm = (volume - v_min) / (v_max - v_min)
    volume_norm = (volume_norm * 255).astype("int")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(shape[-1]):
        abs_path = os.path.join(save_path, str(i)+".png")
        cv.imwrite(abs_path, volume_norm[..., i])


# calculate the cube information
def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    fold = dim / cube_size + ita
    ovlap = np.ceil(
        np.true_divide(
            (fold * cube_size - dim),
            (fold - 1)))  # dim+ita*cubesize-dim
    ovlap = ovlap.astype('int')
    # print( "ovlap:", str( ovlap ) )#[62 62 86]
    fold = np.ceil(np.true_divide((dim + (fold - 1) * ovlap), cube_size))
    fold = fold.astype('int')
    # print( "fold:", str( fold) ) fold: [8 8 6]
    return fold, ovlap


# decompose volume into list of cubes
def decompose_vol2cube_brain(vol_data, cube_size, n_chn, ita):
    cube_list = []
    fold, ovlap = fit_cube_param(vol_data.shape[0:3], cube_size, ita)
    dim = np.asarray(vol_data.shape[0:3])  # [307, 307, 143]
    # decompose
    for R in range(0, fold[0]):
        r_s = R * cube_size - R * ovlap[0]
        r_e = r_s + cube_size
        if r_e >= dim[0]:  # see if exceed the boundry
            r_s = dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C * cube_size - C * ovlap[1]
            c_e = c_s + cube_size
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H * cube_size - H * ovlap[2]
                h_e = h_s + cube_size
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size
                    h_e = h_s + cube_size
                # partition multiple channels
                cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e, :]
                # By default batch_size = 1
                cube_batch = np.zeros(
                    [1, cube_size, cube_size, cube_size, n_chn]).astype('float32')
                cube_batch[0, :, :, :, :] = copy.deepcopy(cube_temp)
                # save
                cube_list.append(cube_batch)

    return cube_list


# compose list of label cubes into a label volume
def compose_label_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (
        np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('int32')
    idx_classes_mat = (
        np.zeros([cube_size, cube_size, cube_size, class_n])).astype('int32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R * cube_size - R * ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C * cube_size - C * ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H * cube_size - H * ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                # histogram for voting (one-hot)
                for k in range(class_n):
                    idx_classes_mat[:, :, :, k] = (cube_list[p_count] == k)
                # accumulation
                label_classes_mat[r_s:r_e,
                                  c_s:c_e,
                                  h_s:h_e,
                                  :] = label_classes_mat[r_s:r_e,
                                                         c_s:c_e,
                                                         h_s:h_e,
                                                         :] + idx_classes_mat

                p_count += 1
    # print 'label mat unique:'
    # print np.unique(label_mat)

    compose_vol = np.argmax(label_classes_mat, axis=3)
    # print np.unique(label_mat)

    return compose_vol


# compose list of probability cubes into a probability volumes
def compose_prob_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    map_classes_mat = (
        np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')
    cnt_classes_mat = (
        np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R * cube_size - R * ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C * cube_size - C * ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H * cube_size - H * ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                # accumulation
                map_classes_mat[r_s:r_e,
                                c_s:c_e,
                                h_s:h_e,
                                :] = map_classes_mat[r_s:r_e,
                                                     c_s:c_e,
                                                     h_s:h_e,
                                                     :] + cube_list[p_count]
                cnt_classes_mat[r_s:r_e,
                                c_s:c_e,
                                h_s:h_e,
                                :] = cnt_classes_mat[r_s:r_e,
                                                     c_s:c_e,
                                                     h_s:h_e,
                                                     :] + 1.0

                p_count += 1

    # elinimate NaN
    nan_idx = (cnt_classes_mat == 0)
    cnt_classes_mat[nan_idx] = 1.0
    # average
    compose_vol = map_classes_mat / cnt_classes_mat

    return compose_vol


# Remove small connected components
def remove_minor_cc(vol_data, rej_ratio, rename_map):
    """Remove small connected components refer to rejection ratio"""
    """Usage
        # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
        # nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_output/test/test_4.nii'
        # vol_file = nib.load(nii_path)
        # vol_data = vol_file.get_data().copy()
        # ref_affine = vol_file.affine
        # rem_vol = remove_minor_cc(vol_data, rej_ratio=0.2, class_n=8, rename_map=rename_map)
        # # save
        # rem_path = 'rem_cc.nii'
        # rem_vol_file = nib.Nifti1Image(rem_vol, ref_affine)
        # nib.save(rem_vol_file, rem_path)

        #===# possible be parallel in future
    """

    rem_vol = copy.deepcopy(vol_data)
    class_n = len(rename_map)
    # retrieve all classes
    for c in range(1, class_n):
        print('processing class %d...' % c)

        class_idx = (vol_data == rename_map[c]) * 1
        class_vol = np.sum(class_idx)
        labeled_cc, num_cc = measurements.label(class_idx)
        # retrieve all connected components in this class
        for cc in range(1, num_cc + 1):
            single_cc = ((labeled_cc == cc) * 1)
            single_vol = np.sum(single_cc)
            # remove if too small
            if single_vol / (class_vol * 1.0) < rej_ratio:
                rem_vol[labeled_cc == cc] = 0

    return rem_vol


def background_num_to_save(input_gt, fg_ratio, bg_ratio):
    background_num = tf.reduce_sum(input_gt[:, :, :, :, 0])
    total_num = tf.reduce_sum(input_gt)
    foreground_num = total_num - background_num
    # save_back_ground_num = tf.reduce_max(
    #     [2 * foreground_num, background_num / 32])  # set the number of background samples to reserve
    save_back_ground_num = tf.reduce_max(
        [fg_ratio * foreground_num, background_num / bg_ratio])  # set the number of background samples to reserve
    save_back_ground_num = tf.clip_by_value(
        save_back_ground_num, 0, background_num)
    return save_back_ground_num


def no_background(input_gt):
    return input_gt


def exist_background(input_gt, pred, save_back_ground_num):
    batch, in_depth, in_height, in_width, in_channels = [
        int(d) for d in input_gt.get_shape()]
    pred_data = pred[:, :, :, :, 0]

    gt_backgound_data = 1 - input_gt[:, :, :, :, 0]
    pred_back_ground_data = tf.reshape(
        pred_data, (batch, in_depth * in_height * in_width))
    gt_back_ground_data = tf.reshape(
        gt_backgound_data,
        (batch,
         in_depth *
         in_height *
         in_width))

    new_pred_data = pred_back_ground_data + gt_back_ground_data

    mask = []
    for i in range(batch):
        gti = -1 * new_pred_data[i, :]
        max_k_number, index = tf.nn.top_k(
            gti, save_back_ground_num)
        max_k = tf.reduce_min(max_k_number)
        one = tf.ones_like(gti)  # all 1 mask
        zero = tf.zeros_like(gti)  # all 0 mask
        mask_slice = tf.where(gti < max_k, x=zero, y=one)
        mask_slice = tf.reshape(mask_slice, [in_depth, in_height, in_width])
        mask.append(mask_slice)
    mask = tf.expand_dims(mask, -1)
    other_mask = tf.ones([batch,
                          in_depth,
                          in_height,
                          in_width,
                          in_channels - 1],
                         tf.float32)
    full_mask = tf.concat([mask, other_mask], 4)

    input_gt = full_mask * input_gt
    return input_gt


# Get a background mask for the groundtruth so that we can
# discard the unnecessary background information
def produce_mask_background(input_gt, pred, fg_ratio, bg_ratio):
    save_back_ground_num = background_num_to_save(
        input_gt, fg_ratio, bg_ratio)  # Get the background numbers to reserve from groundtruth
    save_back_ground_num = tf.cast(
        save_back_ground_num,
        dtype=tf.int32)
    product = tf.cond(
        save_back_ground_num < 5,
        lambda: no_background(input_gt),
        lambda: exist_background(
            input_gt,
            pred,
            save_back_ground_num))

    return product

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def postprocessing(input_volume):
    _,_, slices = input_volume.shape
    volume_out = np.zeros(input_volume.shape, dtype="int16")
    input_volume = input_volume*255
    for i in range(slices):
        temp = fillhole(input_volume[..., i])
        volume_out[:, :, i] = temp
    volume_out = (volume_out/255).astype("int16")
    return volume_out

def majority_voting(array):
    '''
    this function realize the majority voting algorithm.
    :param array: input array need to processed
    :return: majority numbet
    '''
    count = Counter(array)
    majo = count.most_common(1)
    return majo

def multi_majority_voting(ndaray):
    shape = ndaray.shape
    out = np.zeros(shape[0:3])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                array_vote = [ndaray[i,j,k,0],ndaray[i,j,k,1],ndaray[i,j,k,2],ndaray[i,j,k,3],ndaray[i,j,k,4] ]
                out[i,j,k] = majority_voting(array_vote)[0][0]
    return out

def five_fold_validation(dataset, outpath):
    path1 = '/home/server/home/5foldtest/fold1'
    path2 = '/home/server/home/5foldtest/fold2'
    path3 = '/home/server/home/5foldtest/fold3'
    path4 = '/home/server/home/5foldtest/fold4'
    path5 = '/home/server/home/5foldtest/fold5'
    file_list = os.listdir(path1)
    datalist = []
    for file in file_list:
        file_abs_path1 = os.path.join(path1, file)
        volume1 = nib.load(file_abs_path1)
        data1 = volume1.get_data()

        file_abs_path2 = os.path.join(path2, file)
        volume2 = nib.load(file_abs_path2)
        data2 = volume2.get_data()

        file_abs_path1 = os.path.join(path1, file)
        volume1 = nib.load(file_abs_path1)
        data1 = volume1.get_data()

        file_abs_path1 = os.path.join(path1, file)
        volume1 = nib.load(file_abs_path1)
        data1 = volume1.get_data()

        file_abs_path1 = os.path.join(path1, file)
        volume1 = nib.load(file_abs_path1)
        data1 = volume1.get_data()

def load_train_ini(ini_file):
    # initialize
    cf = configparser.ConfigParser()
    cf.read(ini_file, encoding="utf-8-sig")
    # dictionary list
    param_sections = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(phase=cf.get(s[d], "phase"),
                          batch_size=cf.getint(s[d], "batch_size"),
                          inputI_size=cf.getint(s[d], "inputI_size"),
                          inputI_chn=cf.getint(s[d], "inputI_chn"),
                          outputI_size=cf.getint(s[d], "outputI_size"),
                          output_chn=cf.getint(s[d], "output_chn"),
                          rename_map=cf.get(s[d], "rename_map"),
                          resize_r=cf.getfloat(s[d], "resize_r"),
                          traindata_dir=cf.get(s[d], "traindata_dir"),
                          chkpoint_dir=cf.get(s[d], "chkpoint_dir"),
                          learning_rate=cf.getfloat(s[d], "learning_rate"),
                          beta1=cf.getfloat(s[d], "beta1"),
                          epoch=cf.getint(s[d], "epoch"),
                          model_name=cf.get(s[d], "model_name"),
                          save_intval=cf.getint(s[d], "save_intval"),
                          testdata_dir=cf.get(s[d], "testdata_dir"),
                          labeling_dir=cf.get(s[d], "labeling_dir"),
                          ovlp_ita=cf.getint(s[d], "ovlp_ita"),
                          step=cf.getint(s[d], "step"),
                          Stages=cf.getint(s[d], "Stages"),
                          Blocks=cf.getint(s[d], "Blocks"),
                          Columns=cf.getint(s[d], "Columns"),
                          fg_ratio=cf.getfloat(s[d], "fg_ratio"),
                          bg_ratio=cf.getfloat(s[d], "bg_ratio"),
                          focal_loss_flag=cf.getboolean(s[d], "focal_loss_flag"))
        # add to list
        param_sections.append(level_dict)

    return param_sections



if __name__ == '__main__':
    path = '/home/server/home/5foldtest/fold1/validation/BraTS19_UAB_3498_1.nii.gz'
    path2 = '/home/server/home/5foldtest/'
    dfdfd = five_fold_validation(path2, "validation", "")
    arrrr = np.array(dfdfd)
    vol = nib.load(path)
    img = vol.get_data()
    shape = img.shape
    a = [1,2,1,2,3]
    aa = [1,2,1,2,2]
    aaa = np.array(aa)
    tim1 = time.time()
    # ndar = np.random.randint(0,4,size=(240,240,155,5))
    ndar = np.random.randint(0, 4, size=(240, 240, 155, 5))
    out = multi_majority_voting(ndar)
    tim2 = time.time()
    elaps = tim2 - tim1
    b = majority_voting(aaa)