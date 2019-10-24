import copy
import numpy as np

from char_detection.object_point import data_operator as dt_ope
from image_processing import image_proc


class TranslateAugmentation_9case_ThreshScoreWeightedAve:
    """
    基準結果を基に、オブジェクトの存在する最外部を検出。最外部の外側の範囲で画像をシフトさせる。
    シフトさせた画像で得られたヒートマップ、サイズ、オフセットを逆シフトし、平均する。
    """
    def __init__(self, 
                 ratio_shift_to_gap_w, 
                 ratio_shift_to_gap_h,
                 iou_threshold, 
                 score_threshold,
                 score_threshold_for_weight):

        self.RATIO_SHIFT_TO_GAP_W = ratio_shift_to_gap_w
        self.RATIO_SHIFT_TO_GAP_H = ratio_shift_to_gap_h

        self.IOU_THRESHOLD = iou_threshold
        self.SCORE_THRESHOLD = score_threshold
        self.SCORE_THRESHOLD_FOR_WEIGHT = score_threshold_for_weight

        self.SHIFT_UNIT_SIZE = 4

        self.__initilization()

        return

    def __initilization(self):
        return

    def augment_image(self, images, pred_base_heatmaps, pred_base_sizes, pred_base_offsets):
        """
        Args:
            images: shape=(num_data, H, W, 1)
            pred_heatmaps: shape = (num_data, hm_y, hm_x, num_class)
            pred_obj_sizes: shape = (num_data, hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_offsets: shape = (num_data, hm_y, hm_x, num_class * 2), 2=(w, h)
        """
        image_shape = images.shape[1:]
        num_class = pred_base_heatmaps.shape[3]

        # outer most positions : [most left x, most up y, most right x, most bottom y] * sample_num
        outermost_positions = self.__calc_outermost_position(image_shape, num_class, 
                                                             pred_base_heatmaps, 
                                                             pred_base_sizes, 
                                                             pred_base_offsets)

        shifted_imgs = []
        # loop of data
        for img, outmost_posi in zip(images, outermost_positions):
            aug_8img = []
            # have no bbox
            if len(outmost_posi) == 0:
                pass
            else:
                # shift size
                w_shift_1, w_shift_2, h_shift_1, h_shift_2 = self.__calc_shift_size(image_shape, outmost_posi)

                # shift image
                w_sfts, h_sfts = np.meshgrid([w_shift_1, 0, w_shift_2], [h_shift_1, 0, h_shift_2])
                # loop of aug
                for w_sft, h_sft in zip(w_sfts.flatten(), h_sfts.flatten()):
                    # if true, image is not augment (shift=0)
                    if not(w_sft == 0 and h_sft == 0):
                        aug_img = image_proc.ImageProcessing.translate(img, w_sft, h_sft, mode='edge')
                        aug_8img.append(aug_img)

            # append
            shifted_imgs.append(aug_8img)
        shifted_imgs = np.array(shifted_imgs)

        return shifted_imgs

    def integrate_heatmap_size_offset(self, image,
                                      pred_base_heatmap, pred_base_obj_size, pred_base_offset, 
                                      pred_auged_heatmaps, pred_auged_obj_sizes, pred_auged_offsets):
        """
        Args:
            pred_base_heatmaps: shape = (hm_y, hm_x, num_class)
            pred_base_obj_sizes: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_base_offsets: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)

            pred_auged_heatmaps: shape = (num_aug, hm_y, hm_x, num_class)
            pred_auged_obj_sizes: shape = (num_aug, hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_auged_offsets: shape = (num_aug, hm_y, hm_x, num_class * 2), 2=(w, h)
        """

        # inverse shifted
        intg_hms, intg_szs, intg_ofss = self.__inv_shift_heatmap_size_offset(image,
                                                    pred_base_heatmap, pred_base_obj_size, pred_base_offset, 
                                                    pred_auged_heatmaps, pred_auged_obj_sizes, pred_auged_offsets)

        if len(intg_hms) == 0:
            return copy.copy(pred_base_heatmap), copy.copy(pred_base_obj_size), copy.copy(pred_base_offset)
        else:
            # concatenate [base, auged]
            # integrated hm : shape(num_aug+1, H, W, num_class)
            # integrated sz : shape(num_aug+1, H, W, num_class*2)
            # integrated ofs : shape(num_aug+1, H, W, num_class*2)
            intg_hms = np.concatenate([intg_hms, pred_base_heatmap[np.newaxis,:,:,:]], axis=0)
            intg_szs = np.concatenate([intg_szs, pred_base_obj_size[np.newaxis,:,:,:]], axis=0)
            intg_ofss = np.concatenate([intg_ofss, pred_base_offset[np.newaxis,:,:,:]], axis=0)

            # threshold score weight
            # thresh_score_w : shape(num_aug+1, H, W, num_class)
            #thresh_score_w = np.maximum(intg_hms - self.SCORE_THRESHOLD_FOR_WEIGHT, 0) / (1 - self.SCORE_THRESHOLD_FOR_WEIGHT)
            thresh_score_w = np.sign(np.maximum(intg_hms - self.SCORE_THRESHOLD_FOR_WEIGHT, 0)) * intg_hms
            thresh_score_w = thresh_score_w / (np.sum(thresh_score_w, axis=0) + 1e-7)

            # average with weight
            intg_hms = np.sum(thresh_score_w * intg_hms, axis=0)
            intg_szs = np.sum(thresh_score_w * intg_szs, axis=0)
            intg_ofss = np.sum(thresh_score_w * intg_ofss, axis=0)

            return intg_hms, intg_szs, intg_ofss

    def __calc_shift_size(self, image_shape, outmost_posi):
        """
        Args:
            outmost_posi: [most left x, most up y, most right x, most bottom y]
        Returns:
            w_shift_to_right, w_shift_to_left, h_shift_to_down, w_shift_to_up
        """
        # have no bbox
        if len(outmost_posi) == 0:
            w_shift_1 = 0
            w_shift_2 = 0
            h_shift_1 = 0
            h_shift_2 = 0
        else:
            # out of area including object
            left_gap = np.maximum(outmost_posi[0], 0)
            right_gap = np.maximum(image_shape[1] - outmost_posi[2], 0)
            upper_gap = np.maximum(outmost_posi[1], 0)
            bottom_gap = np.maximum(image_shape[0] - outmost_posi[3], 0)
            # shift size
            w_shift_1 = int((left_gap * self.RATIO_SHIFT_TO_GAP_W) // self.SHIFT_UNIT_SIZE) * self.SHIFT_UNIT_SIZE
            w_shift_2 = - int((right_gap * self.RATIO_SHIFT_TO_GAP_W) // self.SHIFT_UNIT_SIZE) * self.SHIFT_UNIT_SIZE
            h_shift_1 = int((upper_gap * self.RATIO_SHIFT_TO_GAP_H) // self.SHIFT_UNIT_SIZE) * self.SHIFT_UNIT_SIZE
            h_shift_2 = - int((bottom_gap * self.RATIO_SHIFT_TO_GAP_H) // self.SHIFT_UNIT_SIZE) * self.SHIFT_UNIT_SIZE

        return w_shift_1, w_shift_2, h_shift_1, h_shift_2

    def __calc_hm_shift_size(self, image_shape, outmost_posi):
        """
        Args:
            outmost_posi: [most left x, most up y, most right x, most bottom y]
        Returns:
            w_hm_shift_to_right, w_hm_shift_to_left, h_hm_shift_to_down, w_hm_shift_to_up
        """
        w_shift_1, w_shift_2, h_shift_1, h_shift_2 = self.__calc_shift_size(image_shape, outmost_posi)

        w_hm_shift_1 = - w_shift_1 // self.SHIFT_UNIT_SIZE
        w_hm_shift_2 = - w_shift_2 // self.SHIFT_UNIT_SIZE
        h_hm_shift_1 = - h_shift_1 // self.SHIFT_UNIT_SIZE
        h_hm_shift_2 = - h_shift_2 // self.SHIFT_UNIT_SIZE

        return w_hm_shift_1, w_hm_shift_2, h_hm_shift_1, h_hm_shift_2

    def __calc_hm_inverse_shift_size(self, heatmap, obj_size, offset, w_shift_size, h_shift_size):
        """
        Args:
            heatmap: shape = (hm_y, hm_x, num_class)
            obj_size: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)
            offset: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)
        """
        shifted_hm = image_proc.ImageProcessing.translate(heatmap, w_shift_size, h_shift_size, mode='constant')
        shifted_sz = image_proc.ImageProcessing.translate(obj_size, w_shift_size, h_shift_size, mode='constant')
        shifted_ofs = image_proc.ImageProcessing.translate(offset, w_shift_size, h_shift_size, mode='constant')

        return shifted_hm, shifted_sz, shifted_ofs

    def __inv_shift_heatmap_size_offset(self, image,
                                        pred_base_heatmap, pred_base_obj_size, pred_base_offset, 
                                        pred_auged_heatmaps, pred_auged_obj_sizes, pred_auged_offsets):
        """
        Args:
            pred_base_heatmaps: shape = (hm_y, hm_x, num_class)
            pred_base_obj_sizes: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_base_offsets: shape = (hm_y, hm_x, num_class * 2), 2=(w, h)

            pred_auged_heatmaps: shape = (num_aug, hm_y, hm_x, num_class)
            pred_auged_obj_sizes: shape = (num_aug, hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_auged_offsets: shape = (num_aug, hm_y, hm_x, num_class * 2), 2=(w, h)
        """
        image_shape = image.shape[0:]
        num_class = pred_base_heatmap.shape[2]

        # outer most positions : [most left x, most up y, most right x, most bottom y] * sample_num
        outmost_posi = self.__calc_outermost_position(image_shape, num_class, 
                                                      np.array([pred_base_heatmap]), 
                                                      np.array([pred_base_obj_size]), 
                                                      np.array([pred_base_offset]))
        outmost_posi = outmost_posi[0]

        # inverse shift size
        w_hm_shift_1, w_hm_shift_2, h_hm_shift_1, h_hm_shift_2 = self.__calc_hm_shift_size(image_shape, outmost_posi)
        
        # shift hm
        w_hm_sfts, h_hm_sfts = np.meshgrid([w_hm_shift_1, 0, w_hm_shift_2], [h_hm_shift_1, 0, h_hm_shift_2])

        # loop of aug
        inv_sft_hms = []
        inv_sft_szs = []
        inv_sft_ofss = []
        idx_aug_hm = 0
        for iaug, (w_hm_sft, h_hm_sft) in enumerate(zip(w_hm_sfts.flatten(), h_hm_sfts.flatten())):
            if not(w_hm_sft == 0 and h_hm_sft == 0):
                inv_sft_hm, inv_sft_sz, inv_sft_ofs = self.__calc_hm_inverse_shift_size(pred_auged_heatmaps[idx_aug_hm], 
                                                                                        pred_auged_obj_sizes[idx_aug_hm], 
                                                                                        pred_auged_offsets[idx_aug_hm],
                                                                                        w_hm_sft,
                                                                                        h_hm_sft)
                inv_sft_hms.append(inv_sft_hm)
                inv_sft_szs.append(inv_sft_sz)
                inv_sft_ofss.append(inv_sft_ofs)

                idx_aug_hm = idx_aug_hm + 1

        inv_sft_hms = np.array(inv_sft_hms)
        inv_sft_szs = np.array(inv_sft_szs)
        inv_sft_ofss = np.array(inv_sft_ofss)

        return inv_sft_hms, inv_sft_szs, inv_sft_ofss

    def __calc_outermost_position(self, image_shape, num_class, 
                                  pred_heatmaps, pred_obj_sizes, pred_offsets):
        """
        Args:
            pred_heatmaps: shape = (num_data, hm_y, hm_x, num_class)
            pred_obj_sizes: shape = (num_data, hm_y, hm_x, num_class * 2), 2=(w, h)
            pred_offsets: shape = (num_data, hm_y, hm_x, num_class * 2), 2=(w, h)

        Returns:
            [most left x, most up y, most right x, most bottom y] * sample_num
        """

        # calc bbox
        converter_cnet_oup = dt_ope.ConvertCenterNetOutput(num_classes=num_class, image_shape=image_shape)
        bbox_uplfs, bbox_sizes = converter_cnet_oup.to_upleft_points_object_sizes(pred_heatmaps, 
                                                                                  pred_obj_sizes, 
                                                                                  pred_offsets,
                                                                                  self.IOU_THRESHOLD,
                                                                                  self.SCORE_THRESHOLD)

        # outer most position
        outermost_positions = self.__outermost_position(bbox_uplfs, bbox_sizes)

        return outermost_positions

    def __outermost_position(self, upleft_points, object_sizes):
        """
        returns:
            [most left x, most up y, most right x, most bottom y] * sample_num
        """

        # [most left x, most up y, most right x, most bottom y] * sample_num
        outermost_positions = []

        for uplf, obj_sz in zip(upleft_points, object_sizes):
            if uplf.shape[1] != 0:
                # all class, all object
                most_left_x = np.min(uplf[:,:,0])
                most_up_y = np.min(uplf[:,:,1])
                most_right_x = np.max(uplf[:,:,0] + obj_sz[:,:,0])
                most_bottom_y = np.max(uplf[:,:,1] + obj_sz[:,:,1])
                outermost_positions.append([most_left_x, most_up_y, most_right_x, most_bottom_y])
            else:
                outermost_positions.append([])

        return outermost_positions


