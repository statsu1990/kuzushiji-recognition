import copy
import numpy as np
from image_processing import image_proc
from kuzushiji_data import visualization as vis

class TranslateAugmentation_9case:
    def __init__(self, image_size_hw, width_shift_range=0.1, height_shift_range=0.1):

        self.IMAGE_SIZE_HW = image_size_hw
        self.WIDTH_SHIFT_RANGE = width_shift_range
        self.HEIGHT_SHIFT_RANGE = height_shift_range
        
        self.__initilization()

        return

    def __initilization(self):
        self.__build_shift_set()
        return
    
    def __build_shift_set(self):
        w_shift_size = int(self.IMAGE_SIZE_HW[1] * self.WIDTH_SHIFT_RANGE)
        h_shift_size = int(self.IMAGE_SIZE_HW[0] * self.HEIGHT_SHIFT_RANGE)

        w_shifts = np.array([w_shift_size, 0, -w_shift_size])
        h_shifts = np.array([h_shift_size, 0, -h_shift_size])

        wh_shift_set = np.meshgrid(w_shifts, h_shifts)
        self.w_shifts = wh_shift_set[0].flatten()
        self.h_shifts = wh_shift_set[1].flatten()

        return

    def augment_image(self, images, other_inputs=None):
        """
        Returns:
            augmented images: [set1, set2, ..., set9], 
                              where set = [[aug image1, ..., aug imageN], [other inp 1, ..., other inp N]]
        """
        auged_images_set = []
        for w_sft, h_sft in zip(self.w_shifts, self.h_shifts):
            auged_imgs = self.__translate(images, w_sft, h_sft)
            if other_inputs is None:
                auged_images_set.append(auged_imgs)
            else:
                auged_images_set.append([auged_imgs, other_inputs])

        return auged_images_set

    def __translate(self, images, w_shift, h_shift):
        """
        Args:
            images: shape=(sample_num, H, W, C)
        """
        image_h = images.shape[1]
        image_w = images.shape[2]

        # padding
        pad_w_0 = (0, 0)
        pad_w_1 = (np.abs(h_shift), np.abs(h_shift))
        pad_w_2 = (np.abs(w_shift), np.abs(w_shift))
        pad_w_3 = (0, 0)
        translated_images = np.pad(images, pad_width=(pad_w_0, pad_w_1, pad_w_2, pad_w_3), mode='edge')

        #
        w_start = np.abs(w_shift) + w_shift
        w_end = np.abs(w_shift) + image_w + w_shift
        h_start = np.abs(h_shift) + h_shift
        h_end = np.abs(h_shift) + image_h + h_shift

        translated_images = translated_images[:, h_start:h_end, w_start:w_end, :]

        return translated_images

class Augmentation_Translate8case_Zoom4case:
    def __init__(self, shift_rate_wh, zoom_rate):
        self.SHIFT_RATE_WH = shift_rate_wh
        self.ZOOM_RATE = zoom_rate

        self.__initilization()

        return

    def __initilization(self):
        return

    def augment_image(self, images, other_inputs=None):
        """
        Returns:
            augmented images: [set1, set2, ..., set9], 
                              where set = [[aug image1, ..., aug imageN], [other inp 1, ..., other inp N]]
        """
        auged_images_set = []
        
        # original
        if other_inputs is None:
            auged_images_set.append(copy.copy(images))
        else:
            auged_images_set.append([copy.copy(images), copy.copy(other_inputs)])

        #for i in range(3):
        #    vis.Visualization.visualize_gray_img(images[i])

        # translate
        if self.SHIFT_RATE_WH is not None:
            #print('translate')
            tr_imgs_set = self.__translate_8case(images, self.SHIFT_RATE_WH)
            for tr_imgs in tr_imgs_set:
                #for i in range(3):
                #    vis.Visualization.visualize_gray_img(tr_imgs[i])

                if other_inputs is None:
                    auged_images_set.append(tr_imgs)
                else:            
                    auged_images_set.append([tr_imgs, copy.copy(other_inputs)])

        # zoom
        if self.ZOOM_RATE is not None:
            #print('zoom')
            zoom_imgs_set = self.__zoom(images, self.ZOOM_RATE, num_case=4)
            for zm_imgs in zoom_imgs_set:
                #for i in range(3):
                #    vis.Visualization.visualize_gray_img(zm_imgs[i])

                if other_inputs is None:
                    auged_images_set.append(zm_imgs)
                else:            
                    auged_images_set.append([zm_imgs, copy.copy(other_inputs)])

        return auged_images_set

    def __translate_8case(self, images, shift_rate_wh):
        """
        Returns:
            tr_imgs_set: list [tr_imgs1(ndarray), ..., tr_imgs8(ndarray)]
        """
        # shift size
        w_shift = int(images.shape[2] * shift_rate_wh[0])
        h_shift = int(images.shape[1] * shift_rate_wh[1])

        # shift size set
        wh_shift_set = np.meshgrid(np.array([w_shift, 0, -w_shift]), 
                                   np.array([h_shift, 0, -h_shift]))
        w_shifts = wh_shift_set[0].flatten()
        h_shifts = wh_shift_set[1].flatten()

        # remove 0 shift
        not_0sft = np.logical_or(w_shifts!=0, h_shifts!=0)
        w_shifts = w_shifts[not_0sft]
        h_shifts = h_shifts[not_0sft]

        # [tr_imgs1, ..., tr_imgs8]
        tr_imgs_set = []
        for w_sft, h_sft in zip(w_shifts, h_shifts):
            # tr_imgs: shape(num_sample, H, W, C)
            tr_imgs = image_proc.ImageProcessing.translate(images, w_sft, h_sft, mode='edge')
            tr_imgs_set.append(tr_imgs)

        return tr_imgs_set

    def __zoom(self, images, zoom_rate, num_case):
        """
        Returns:
            zoom_imgs_set: list [zoom_imgs_1(ndarray), ..., zoom_imgs_num_case(ndarray)]
        """
        #zoom_rates = [zoom_rate / (num_case/2) * icase for icase in range(-int(num_case/2), int(num_case/2) + 1)]
        #zoom_rates = [zm for zm in zoom_rates if zm != 0]
        zoom_rates = [-0.1, -0.05]

        zoom_imgs_set = []
        for zm_rate in zoom_rates:
            zoom_imgs = image_proc.ImageProcessing.zoom(images, zm_rate, interpolation='INTER_LINEAR')
            zoom_imgs_set.append(zoom_imgs)

        return zoom_imgs_set


