import copy
import numpy as np
from keras.utils import Sequence
from skimage import transform
from skimage import util as ski_util
import cv2

from char_detection.object_point import util as op_util
from kuzushiji_data import visualization as visu

class CenterNetDataGenerator(Sequence):
    """
    reference:
        https://www.kumilog.net/entry/numpy-data-augmentation
        https://www.kumilog.net/entry/keras-generator
    """
    def __init__(self, class_num, image, upleft_points, object_sizes, 
                       batch_size,
                       shift_brightness_range=None,
                       do_shift_width_height=False, 
                       crop_cut_rate=None,
                       zoom_out_rate=None,
                       zoom_rate=None,
                       random_erasing_kwargs=None):
        """
        Args:
            shift_brightness_range : range = (min, max)
            zoom_out_rate: zoom_out_rate >= 0. zoom out image size = original size * (1 + zoom_out_rate)
        """
        # data
        self.CLASS_NUM = class_num
        self.IMAGE = image
        self.UPLEFT_POINTS = upleft_points
        self.OBJECT_SIZES = object_sizes
        
        # batch size
        self.BATCH_SIZE = batch_size

        # augumentation
        self.SHIFT_BRIGHTNESS_RANGE = shift_brightness_range
        self.DO_SHIFT_WIDTH_HEIGHT = do_shift_width_height
        self.CROP_CUT_RATE = crop_cut_rate
        self.ZOOM_OUT_RATE = zoom_out_rate
        self.ZOOM_RATE = zoom_rate

        # {'erasing_prob':, 'area_rate_low':, 'area_rate_high':, 'aspect_rate_low':, 'aspect_rate_high':}
        self.RANDOM_ERASING_KWARGS = random_erasing_kwargs

        self.CALC_HM_EACH_TIME = self.DO_SHIFT_WIDTH_HEIGHT or (self.CROP_CUT_RATE is not None) or (self.ZOOM_OUT_RATE is not None) or (self.ZOOM_RATE is not None) or (self.RANDOM_ERASING_KWARGS is not None)

        # initilize
        self.__initialize()
        
        return

    def __initialize(self):
        self.SAMPLE_NUM = len(self.IMAGE)
        self.STEP_NUM = int(np.ceil(self.SAMPLE_NUM / self.BATCH_SIZE))

        self.cnet_data = CenterNetData(num_classes=self.CLASS_NUM, image_shape=self.IMAGE[0].shape)

        if not self.CALC_HM_EACH_TIME:
            self.heatmap_objsize_offset = self.cnet_data.calc_heatmap_size_offset_concatenated(self.UPLEFT_POINTS, self.OBJECT_SIZES)
        else:
            self.outermost_positions = self.__outermost_position(self.UPLEFT_POINTS, self.OBJECT_SIZES)

        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)

        return

    def __getitem__(self, idx):
        return self.calc_auged_img_heatmap_size_offset(idx)

    def __len__(self):
        return self.STEP_NUM

    def on_epoch_end(self):
        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)
        return

    def calc_auged_img_heatmap_size_offset(self, idx):
        batch_idxes = self.idxes[idx * self.BATCH_SIZE : (idx + 1) * self.BATCH_SIZE]

        # get batch sampe
        img = self.IMAGE[batch_idxes]

        # augment gray brightness shift
        if self.SHIFT_BRIGHTNESS_RANGE is not None:
            img = self.__augment_gray_brightness_shift(img)
                
        # batch of uppper left point and object size
        if self.CALC_HM_EACH_TIME:
            uplf = [copy.copy(self.UPLEFT_POINTS[ii]) for ii in batch_idxes]
            objsz = [copy.copy(self.OBJECT_SIZES[ii]) for ii in batch_idxes]

        # augment zoomout
        if self.ZOOM_OUT_RATE is not None:
            img, uplf, objsz = self.__augment_zoom_out(img, uplf, objsz)

        # augment zoom
        if self.ZOOM_RATE is not None:
            outmost = self.__outermost_position(uplf, objsz)
            img, uplf, objsz = self.__augment_zoom(img, uplf, objsz, outmost)

        # augment width and hight shift
        if self.DO_SHIFT_WIDTH_HEIGHT:
            outmost = [self.outermost_positions[ii] for ii in batch_idxes]
            img, uplf = self.__augment_width_height_shift(img, uplf, outmost)
                
        # augment crop
        if self.CROP_CUT_RATE is not None:
            outmost = self.__outermost_position(uplf, objsz)
            img, uplf, objsz = self.__augment_crop(img, uplf, objsz, outmost)

        # random erasing
        if self.RANDOM_ERASING_KWARGS is not None:
            #img = self.__random_erasing(img, uplf, objsz, **self.RANDOM_ERASING_KWARGS)
            img = self.__random_erasing2(img, uplf, objsz, **self.RANDOM_ERASING_KWARGS)

        # calc y
        if self.CALC_HM_EACH_TIME:
            y = self.cnet_data.calc_heatmap_size_offset_concatenated(uplf, objsz, do_print=False)
        else:
            # get batch sampe
            y = self.heatmap_objsize_offset[batch_idxes]

        # test
        #for itest in range(20):
            #visu.Visualization.visualize_gray_img(img[itest] * 127.5 + 127.5)
            #visu.Visualization.visualize_pred_result(img[itest] * 127.5 + 127.5, uplf[itest][0], objsz[itest][0])

        return img, y

    def __get_indexes(self, sample_num, do_shuffle=True):
        '''
        return shuffled indexes.
        '''
        indexes = np.arange(sample_num)
        if do_shuffle:
            indexes = np.random.permutation(indexes)
        return indexes

    def __augment_gray_brightness_shift(self, image):
        sample_num = len(image)

        shift = np.random.rand(sample_num) * (self.SHIFT_BRIGHTNESS_RANGE[1] - self.SHIFT_BRIGHTNESS_RANGE[0]) + self.SHIFT_BRIGHTNESS_RANGE[0]
        shift = shift[:,np.newaxis,np.newaxis,np.newaxis]
        auged_image = image + shift # [-1, 1]にクリップしたほうがいい？

        return auged_image

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

    def __augment_width_height_shift(self, images, upleft_points, outermost_positions):
        """
        Shift so that the outermost position does not go out of the image.

        Args:
            outermost_positions: [most left x, most up y, most right x, most bottom y] * sample_num
        Returns:
            shifted_images:
            shifted_upleft_points: [x, y] * num_object * num_class * num_sample
        """
        image_h = images.shape[1]
        image_w = images.shape[2]

        shifted_images = []
        shifted_upleft_points = []
        for img, uplf, outermost in zip(images, upleft_points, outermost_positions):
            if len(outermost) != 0:
                # shift size
                w_shift, h_shift = self.__calc_w_h_shift_size(outermost, image_w, image_h)

                # shift image
                shifted_img = self.__translate_image_nearfill(img, w_shift, h_shift)
                shifted_images.append(shifted_img)

                # shift upleft point
                shifted_uplf = self.__calc_shifted_upleft_of_one_data(uplf, w_shift, h_shift)
                shifted_upleft_points.append(shifted_uplf)
            else:
                shifted_images.append(np.copy(img))
                shifted_upleft_points.append(copy.copy(uplf))

        #
        shifted_images = np.array(shifted_images)

        return shifted_images, shifted_upleft_points

    def __calc_w_h_shift_size(self, outermost, image_w, image_h):
        """
        Args:
            outermost : [most left x, most up y, most right x, most bottom y] * sample_num
        """
        w_shift_max = np.maximum(int(outermost[0]), 0)
        w_shift_min = np.minimum(- int(image_w - outermost[2]), 0)
        w_shift = self.__randint(w_shift_min, w_shift_max)

        h_shift_max = np.maximum(int(outermost[1]), 0)
        h_shift_min = np.minimum(- int(image_h - outermost[3]), 0)
        h_shift = self.__randint(h_shift_min, h_shift_max)

        return w_shift, h_shift

    def __translate_image_nearfill(self, image, w_shift_size, h_shift_size):
        """
        translate image.
        Args:
            image: shape=(h,w,c)
        Returns:
            translated image
        """
        #matrix_trans = transform.AffineTransform(translation=(w_shift_size, h_shift_size))
        #trans_img = transform.warp(image, matrix_trans, mode='edge', preserve_range=True)

        # https://docs.opencv.org/3.4.1/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
        # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        w, h, c = image.shape
        matrix_trans = np.float32([[1, 0, -w_shift_size], [0, 1, -h_shift_size]])
        trans_img = cv2.warpAffine(image, matrix_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if len(trans_img.shape) == 2:
            trans_img = trans_img[:,:,np.newaxis]

        return trans_img

    def __augment_crop(self, images, upleft_points, object_sizes, outermost_positions):
        """
        Args:
            outermost_positions: [most left x, most up y, most right x, most bottom y] * sample_num
        Returns:
            croped images:
            croped upleft_points: [x, y] * num_object * num_class * num_sample
            croped object_sizes: [x, y] * num_object * num_class * num_sample
        """
        image_h = images.shape[1]
        image_w = images.shape[2]

        croped_images = []
        croped_upleft_points = []
        croped_object_sizes = []
        for img, uplf, objsz, outermost in zip(images, upleft_points, object_sizes, outermost_positions):
            if len(outermost) != 0:
                # cut position
                cut = int(np.minimum((int(outermost[0]) - 0) / image_w, self.CROP_CUT_RATE) * image_w)
                cut = np.maximum(cut, 0)
                w_cut_left = self.__randint(0, 0 + cut)

                cut = int(np.minimum((image_w - int(outermost[2])) / image_w, self.CROP_CUT_RATE) * image_w)
                cut = np.maximum(cut, 0)
                w_cut_right = self.__randint(image_w - cut, image_w)
                
                cut = int(np.minimum((int(outermost[1]) - 0) / image_h, self.CROP_CUT_RATE) * image_h)
                cut = np.maximum(cut, 0)
                h_cut_up = self.__randint(0, 0 + cut)

                cut = int(np.minimum((image_h - int(outermost[3])) / image_h, self.CROP_CUT_RATE) * image_h)
                cut = np.maximum(cut, 0)
                h_cut_bottom = self.__randint(image_h - cut, image_h)

                # image
                # crop
                croped_img = img[h_cut_up:h_cut_bottom, w_cut_left:w_cut_right]
                croped_h = croped_img.shape[0]
                croped_w = croped_img.shape[1]
                # resize
                #croped_img = transform.resize(croped_img, output_shape=(image_h, image_w), preserve_range=True)
                croped_img = cv2.resize(croped_img, dsize=(image_h, image_w))
                if len(croped_img.shape) == 2:
                    croped_img = croped_img[:,:,np.newaxis]
                croped_images.append(croped_img)

                # upleft point
                croped_uplf = self.__calc_shifted_upleft_of_one_data(uplf, w_cut_left, h_cut_up)
                croped_uplf = self.__calc_resized_upleft_of_one_data(croped_uplf, croped_w, croped_h, image_w, image_h)
                croped_upleft_points.append(croped_uplf)

                # object size
                croped_objsz = self.__calc_resizeed_obj_size_of_one_data(objsz, croped_w, croped_h, image_w, image_h)
                croped_object_sizes.append(croped_objsz)
            else:
                croped_images.append(np.copy(img))
                croped_upleft_points.append(copy.copy(uplf))
                croped_object_sizes.append(copy.copy(objsz))

        croped_images = np.array(croped_images)
        
        return croped_images, croped_upleft_points, croped_object_sizes

    def __augment_zoom_out(self, images, upleft_points, object_sizes):

        image_h = images.shape[1]
        image_w = images.shape[2]

        max_pad_w = int(image_w * self.ZOOM_OUT_RATE / 2)
        max_pad_h = int(image_h * self.ZOOM_OUT_RATE / 2)

        zoomout_images = []
        zoomout_upleft_points = []
        zoomout_object_sizes = []
        for img, uplf, objsz in zip(images, upleft_points, object_sizes):
            # pad
            pad_w_left = self.__randint(0, max_pad_w)
            pad_w_right = self.__randint(0, max_pad_w)
            pad_h_up = self.__randint(0, max_pad_h)
            pad_h_bottom = self.__randint(0, max_pad_h)

            # image
            # zoom out
            #zoomout_img = ski_util.pad(img, pad_width=((pad_h_up, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)), mode='edge')
            zoomout_img = np.pad(img, pad_width=((pad_h_up, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)), mode='edge')
            zoomout_h = zoomout_img.shape[0]
            zoomout_w = zoomout_img.shape[1]
            # resize
            #zoomout_img = transform.resize(zoomout_img, output_shape=(image_h, image_w), preserve_range=True)
            zoomout_img = cv2.resize(zoomout_img, dsize=(image_h, image_w))
            if len(zoomout_img.shape) == 2:
                zoomout_img = zoomout_img[:,:,np.newaxis]
            zoomout_images.append(zoomout_img)

            # upleft point
            zoomout_uplf = self.__calc_shifted_upleft_of_one_data(uplf, -pad_w_left, -pad_h_up)
            zoomout_uplf = self.__calc_resized_upleft_of_one_data(zoomout_uplf, zoomout_w, zoomout_h, image_w, image_h)
            zoomout_upleft_points.append(zoomout_uplf)
            
            # object size
            zoomout_objsz = self.__calc_resizeed_obj_size_of_one_data(objsz, zoomout_w, zoomout_h, image_w, image_h)
            zoomout_object_sizes.append(zoomout_objsz)

        zoomout_images = np.array(zoomout_images)

        return zoomout_images, zoomout_upleft_points, zoomout_object_sizes

    def __augment_zoom(self, images, upleft_points, object_sizes, outermost_positions):
        """
        Args:
            outermost_positions: [most left x, most up y, most right x, most bottom y] * sample_num
        Returns:
            zoomed images:
            zoomed upleft_points: [x, y] * num_object * num_class * num_sample
            zoomed object_sizes: [x, y] * num_object * num_class * num_sample
        """
        image_h = images.shape[1]
        image_w = images.shape[2]

        zoom_images = []
        zoom_upleft_points = []
        zoom_object_sizes = []
        for img, uplf, objsz, outermost in zip(images, upleft_points, object_sizes, outermost_positions):
            if len(outermost) != 0:
                # cut position
                cut = int(np.minimum((int(outermost[0]) - 0) / image_w, self.ZOOM_RATE) * image_w)
                cut = np.maximum(cut, 0)
                w_cut_left = self.__randint(-cut, cut)

                cut = int(np.minimum((image_w - int(outermost[2])) / image_w, self.ZOOM_RATE) * image_w)
                cut = np.maximum(cut, 0)
                w_cut_right = self.__randint(-cut, cut)
                
                cut = int(np.minimum((int(outermost[1]) - 0) / image_h, self.ZOOM_RATE) * image_h)
                cut = np.maximum(cut, 0)
                h_cut_up = self.__randint(-cut, cut)

                cut = int(np.minimum((image_h - int(outermost[3])) / image_h, self.ZOOM_RATE) * image_h)
                cut = np.maximum(cut, 0)
                h_cut_bottom = self.__randint(-cut, cut)

                # image
                zoom_img = np.pad(img, 
                                  pad_width=((np.abs(h_cut_up), np.abs(h_cut_bottom)), 
                                             (np.abs(w_cut_left), np.abs(w_cut_right)), 
                                             (0, 0)), 
                                  mode='edge')
                # zoom
                zoom_img = zoom_img[np.abs(h_cut_up) + h_cut_up : np.abs(h_cut_up) + image_h - h_cut_bottom, 
                                    np.abs(w_cut_left) + w_cut_left : np.abs(w_cut_left) + image_w - w_cut_right, 
                                    :]
                zoom_h = zoom_img.shape[0]
                zoom_w = zoom_img.shape[1]
                # resize
                zoom_img = cv2.resize(zoom_img, dsize=(image_h, image_w))
                if len(zoom_img.shape) == 2:
                    zoom_img = zoom_img[:,:,np.newaxis]
                zoom_images.append(zoom_img)

                # upleft point
                zoom_uplf = self.__calc_shifted_upleft_of_one_data(uplf, w_cut_left, h_cut_up)
                zoom_uplf = self.__calc_resized_upleft_of_one_data(zoom_uplf, zoom_w, zoom_h, image_w, image_h)
                zoom_upleft_points.append(zoom_uplf)

                # object size
                zoom_objsz = self.__calc_resizeed_obj_size_of_one_data(objsz, zoom_w, zoom_h, image_w, image_h)
                zoom_object_sizes.append(zoom_objsz)
            else:
                zoom_images.append(np.copy(img))
                zoom_upleft_points.append(copy.copy(uplf))
                zoom_object_sizes.append(copy.copy(objsz))

        zoom_images = np.array(zoom_images)

        return zoom_images, zoom_upleft_points, zoom_object_sizes

    def __random_erasing(self, images, upleft_points, object_sizes, 
                         erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        target_images = []

        for img, uplfs, obj_szs in zip(images, upleft_points, object_sizes):
            target_image = self.__random_erasing_one_image(img, uplfs, obj_szs, 
                                        erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high)
            target_images.append(target_image)

        target_images = np.array(target_images)

        return target_images

    def __random_erasing2(self, images, upleft_points, object_sizes, 
                         erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        target_images = []

        for img, uplfs, obj_szs in zip(images, upleft_points, object_sizes):
            target_image = copy.copy(img)
            target_image = self.__random_erasing_one_image2(target_image, uplfs, obj_szs, 
                                        erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high, 
                                        bbox_aware=True)
            target_image = self.__random_erasing_one_image2(target_image, uplfs, obj_szs, 
                                        erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high, 
                                        bbox_aware=False)
            target_images.append(target_image)

        target_images = np.array(target_images)

        return target_images

    def __random_erasing_one_image2(self, image, upleft_points, object_sizes, 
                                   erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high, bbox_aware=True):
        """
        Args:
            upleft_points: shape = (num_class, num_bbox, 2)
        """

        # concatenate
        for icls, (uplfs_cls, obj_szs_cls) in enumerate(zip(upleft_points, object_sizes)):
            if icls == 0:
                uplfs = copy.copy(uplfs_cls)
                obj_szs = copy.copy(obj_szs_cls)
            else:
                uplfs = np.concatenate([uplfs, uplfs_cls], axis=0)
                obj_szs = np.concatenate([obj_szs, obj_szs_cls], axis=0)

        # have bbox?
        if len(uplfs) == 0:
            erased_img = copy.copy(image)
        else:
            # target bbox
            target_idxs = np.arange(len(uplfs))[np.random.rand(len(uplfs)) < erasing_prob]

            # have target?
            if len(target_idxs) == 0:
                erased_img = copy.copy(image)
            else:
                uplfs = uplfs[target_idxs]
                obj_szs = obj_szs[target_idxs]

                num_bbox = len(uplfs)

                # base size
                Hs = np.maximum(obj_szs[:,1], 1)
                Ws = np.maximum(obj_szs[:,0], 1)
                Ss = Hs * Ws

                # image size
                H_img = image.shape[0]
                W_img = image.shape[1]
                C = image.shape[2]

                # erasing size
                Ss_er = np.random.uniform(area_rate_low, area_rate_high, num_bbox) * Ss # 画像に重畳する矩形の面積
                rs_er = np.random.uniform(aspect_rate_low, aspect_rate_high, num_bbox) # 画像に重畳する矩形のアスペクト比

                Hs_er = np.maximum(np.sqrt(Ss_er * rs_er).astype('int'), 1) # 画像に重畳する矩形のHeight
                Ws_er = np.maximum(np.sqrt(Ss_er / rs_er).astype('int'), 1) # 画像に重畳する矩形のWidth

                # erasing position
                if bbox_aware:
                    W_range = Ws
                    H_range = Hs
                else:
                    W_range = W_img
                    H_range = H_img

                center_xs_er = ((1.0 - np.minimum(Ws_er*0.5/W_range, 0.4) * 2) * np.random.rand(num_bbox) + np.minimum(Ws_er*0.5/W_range, 0.4)) * W_range
                center_xs_er = np.ceil(center_xs_er)
                center_ys_er = ((1.0 - np.minimum(Hs_er*0.5/H_range, 0.4) * 2) * np.random.rand(num_bbox) + np.minimum(Hs_er*0.5/H_range, 0.4)) * H_range
                center_ys_er = np.ceil(center_ys_er)

                if bbox_aware:
                    center_xs_er = uplfs[:,0] + center_xs_er
                    center_ys_er = uplfs[:,1] + center_ys_er

                xs1_er = np.clip((center_xs_er - 0.5 * Ws_er).astype('int'), 0, W_img)
                ys1_er = np.clip((center_ys_er - 0.5 * Hs_er).astype('int'), 0, H_img)
                xs2_er = np.clip((center_xs_er + 0.5 * Ws_er).astype('int'), 0, W_img)
                ys2_er = np.clip((center_ys_er + 0.5 * Hs_er).astype('int'), 0, H_img)
            
                # mask
                mask = np.zeros((H_img, W_img, 1))
                for x1_er, y1_er, x2_er, y2_er in zip(xs1_er, ys1_er, xs2_er, ys2_er):
                    mask[y1_er:y2_er, x1_er:x2_er, :] = 1

                # random noise
                noise = np.random.uniform(np.min(image), np.max(image), (H_img, W_img, C))

                #
                erased_img = image * (1 - mask) + noise * mask

        return erased_img

    def __random_erasing_one_image(self, image, upleft_points, object_sizes, 
                                   erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        """
        Args:
            upleft_points: shape = (num_class, num_bbox, 2)
        """
        target_image = image.copy()

        for uplf_cls, obj_sz_cls in zip(upleft_points, object_sizes):
            for uplf_box, obj_sz_box in zip(uplf_cls, obj_sz_cls):
                if len(uplf_box) > 0:
                    target_image = self.__random_erasing_one_bbox(target_image, uplf_box, obj_sz_box, 
                                                erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high)
                    target_image = self.__random_erasing_one_bbox_image_aware(target_image, uplf_box, obj_sz_box, 
                                                erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high)

        return target_image

    def __random_erasing_one_bbox(self, image, upleft_point, object_size, 
                                   erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        target_image = image.copy()

        if np.random.rand() > erasing_prob:
            # RandomErasingを実行しない
            return target_image 

        #H, W, C = target_image.shape
        H = np.maximum(object_size[1], 1)
        W = np.maximum(object_size[0], 1)
        C = target_image.shape[2]

        S = H * W

        #while True:
        Se = np.random.uniform(area_rate_low, area_rate_high) * S # 画像に重畳する矩形の面積
        re = np.random.uniform(aspect_rate_low, aspect_rate_high) # 画像に重畳する矩形のアスペクト比

        He = np.maximum(int(np.sqrt(Se * re)), 1) # 画像に重畳する矩形のHeight
        We = np.maximum(int(np.sqrt(Se / re)), 1) # 画像に重畳する矩形のWidth

        if We/4 + 1 < W - We/4:
            xe = np.random.randint(We/4, W - We/4) # 画像に重畳する矩形のx座標
        else:
            xe = np.random.randint(0, W) # 画像に重畳する矩形のx座標
        if He/4 + 1 < H - He/4:
            ye = np.random.randint(He/4, H - He/4) # 画像に重畳する矩形のy座標
        else:
            ye = np.random.randint(0, H) # 画像に重畳する矩形のy座標

        #xe = np.random.randint(0, W) # 画像に重畳する矩形のx座標
        #ye = np.random.randint(0, H) # 画像に重畳する矩形のy座標

        #if xe + We <= W and ye + He <= H:
        #    # 画像に重畳する矩形が画像からはみ出していなければbreak
        #    break

        #mask = np.random.randint(np.min(target_image), np.max(target_image), (He, We, C)) # 矩形がを生成 矩形内の値はランダム値
        #mask = np.random.rand(He, We, C) # 矩形がを生成 矩形内の値はランダム値
        #mask = mask * np.max(target_image) + (1 - mask) * np.min(target_image)
        
        #target_image[ye:ye + He, xe:xe + We, :] = mask # 画像に矩形を重畳

        mask_uplf_x = np.maximum(0, int(upleft_point[0] + xe - We/2))
        mask_uplf_y = np.maximum(0, int(upleft_point[1] + ye - He/2))
        mask_bottomright_x = np.minimum(target_image.shape[1], int(upleft_point[0] + xe + We/2))
        mask_bottomright_y = np.minimum(target_image.shape[0], int(upleft_point[1] + ye + He/2))

        mask = np.random.rand(mask_bottomright_y - mask_uplf_y, mask_bottomright_x - mask_uplf_x, C)
        mask = mask * np.max(target_image) + (1 - mask) * np.min(target_image)

        target_image[mask_uplf_y:mask_bottomright_y, mask_uplf_x:mask_bottomright_x, :] = mask # 画像に矩形を重畳

        return target_image

    def __random_erasing_one_bbox_image_aware(self, image, upleft_point, object_size, 
                                   erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        target_image = image.copy()

        if np.random.rand() > erasing_prob:
            # RandomErasingを実行しない
            return target_image 

        #H, W, C = target_image.shape
        H = np.maximum(object_size[1], 1)
        W = np.maximum(object_size[0], 1)
        C = target_image.shape[2]

        imgH = target_image.shape[0]
        imgW = target_image.shape[1]

        S = H * W

        while True:
            Se = np.random.uniform(area_rate_low, area_rate_high) * S # 画像に重畳する矩形の面積
            re = np.random.uniform(aspect_rate_low, aspect_rate_high) # 画像に重畳する矩形のアスペクト比

            He = int(np.sqrt(Se * re)) # 画像に重畳する矩形のHeight
            We = int(np.sqrt(Se / re)) # 画像に重畳する矩形のWidth

            xe = np.random.randint(0, imgW) # 画像に重畳する矩形のx座標
            ye = np.random.randint(0, imgH) # 画像に重畳する矩形のy座標

            if xe + We <= imgW and ye + He <= imgH:
                # 画像に重畳する矩形が画像からはみ出していなければbreak
                break

        #mask = np.random.randint(np.min(target_image), np.max(target_image), (He, We, C)) # 矩形がを生成 矩形内の値はランダム値
        mask = np.random.rand(He, We, C) # 矩形がを生成 矩形内の値はランダム値
        mask = mask * np.max(target_image) + (1 - mask) * np.min(target_image)
            
        target_image[ye:ye + He, xe:xe + We, :] = mask # 画像に矩形を重畳

        return target_image


    def __randint(self, low, high):
        if low == high:
            rint = low
        else:
            rint = np.random.randint(low, high)
        return rint

    def __calc_shifted_upleft_of_one_data(self, upleft_point, shift_w, shift_h):
        shifted_upleft_point = []
        # loop of class
        for uplf_cls in upleft_point:
            # no upper left point
            if len(uplf_cls) == 0:
                shifted_upleft_point.append(copy.copy(uplf_cls))
            else:
                shifted_uplf_cls = uplf_cls - np.array([shift_w, shift_h])
                shifted_upleft_point.append(shifted_uplf_cls)

        shifted_upleft_point = np.array(shifted_upleft_point)

        return shifted_upleft_point

    def __calc_resized_upleft_of_one_data(self, upleft_point, before_w, before_h, after_w, after_h):
        resized_upleft_point = []
        # loop of class
        for uplf_cls in upleft_point:
            # no upper left point
            if len(uplf_cls) == 0:
                resized_upleft_point.append(copy.copy(uplf_cls))
            else:
                resized_uplf_cls = uplf_cls * np.array([after_w/before_w, after_h/before_h])
                resized_upleft_point.append(resized_uplf_cls)

        resized_upleft_point = np.array(resized_upleft_point)

        return resized_upleft_point

    def __calc_resizeed_obj_size_of_one_data(self, obj_size, before_w, before_h, after_w, after_h):
        resized_obj_size = []
        # loop of class
        for objsz_cls in obj_size:
            # no upper left point
            if len(objsz_cls) == 0:
                resized_obj_size.append(copy.copy(objsz_cls))
            else:
                resized_objsz_cls = objsz_cls * np.array([after_w/before_w, after_h/before_h])
                resized_obj_size.append(resized_objsz_cls)

        resized_obj_size = np.array(resized_obj_size)

        return resized_obj_size

class CenterNetData:
    def __init__(self, num_classes, image_shape):
        """
        image_shape : rgb=(y,x,3), gray=(y,x,1)
        """
        self.NUM_CLASSES = num_classes
        self.IMAGE_SHAPE = image_shape
        
        # heat map size
        self.heatmap_size = self.__calc_heatmap_size(self.IMAGE_SHAPE[0:2])
        self.to_heatmap_scale = np.array([self.heatmap_size[0] / self.IMAGE_SHAPE[0], self.heatmap_size[1] / self.IMAGE_SHAPE[1]])

        return

    def __calc_heatmap_size(self, input_size):
        """
        input becomes 1/4 in two (2,2) stride operation.
        """
        ndarr_input_size = np.array(input_size, dtype='int')
        hm_size = (ndarr_input_size + ndarr_input_size % 2) // 2
        hm_size = (hm_size + hm_size % 2) // 2
        return tuple(hm_size.tolist())

    def calc_heatmap_size_offset_concatenated(self, upleft_points, object_sizes, do_print=True):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        
        return
            concatenate([heatmaps, object sizes, offsets])
                where
                def SHAPE(N) = (num_data, hm_y, hm_x, N)

                heatmaps : shape = SHAPE(num_classes), [[[[hm], ...], ...], ...]
                object sizes (width,height) in image scale : shape = SHAPE(2 * num_classes),
                                                                [[[[w, h, w, h, ...], ...], ...], ...]
                offsets (x,y) in heatmap scale : shape = SHAPE(2 * num_classes),
                                                    [[[[ofs_x, ofs_y, ofs_x, ofs_y, ...], ...], ...], ...]
        """
        data_num = len(upleft_points)

        # center net data
        concat_hms_sizes_offsets = []
        if do_print:
            print('calculate heatmap size offset for centernet')
        for idata, (uplf_pt, obj_sz) in enumerate(zip(upleft_points, object_sizes)):
            if do_print and (idata+1) % 100 == 0:
                print('\r {0}/{1} calculated'.format(idata + 1, data_num), end="")

            heatmap, obj_size_map_to_hm, offset_hm = self.calc_heatmap_size_offset_one_data(uplf_pt, obj_sz)
            concat_hms_sizes_offsets.append(np.concatenate([heatmap, obj_size_map_to_hm, offset_hm], axis=-1))
        if do_print:
            print()
        
        # to ndarray
        concat_hms_sizes_offsets = np.array(concat_hms_sizes_offsets)
        
        return concat_hms_sizes_offsets

    def calc_heatmap_size_offset(self, upleft_points, object_sizes):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        
        return
            def SHAPE(N) = (num_data, hm_y, hm_x, N)

            heatmaps : shape = SHAPE(num_classes), [[[[hm], ...], ...], ...]
            object sizes (width,height) in image scale : shape = SHAPE(2 * num_classes),
                                                         [[[[w, h, w, h, ...], ...], ...], ...]
            offsets (x,y) in heatmap scale : shape = SHAPE(2 * num_classes),
                                             [[[[ofs_x, ofs_y, ofs_x, ofs_y, ...], ...], ...], ...]
        """
        data_num = len(upleft_points)

        # center net data
        heatmaps = []
        obj_sizes_map_to_hm = []
        offsets_hm = []
        print('calculate heatmap size offset for centernet')
        for idata, (uplf_pt, obj_sz) in enumerate(zip(upleft_points, object_sizes)):
            if (idata+1) % 100 == 0:
                print('\r {0}/{1} calculated'.format(idata + 1, data_num), end="")

            heatmap, obj_size_map_to_hm, offset_hm = self.calc_heatmap_size_offset_one_data(uplf_pt, obj_sz)
            heatmaps.append(heatmap)
            obj_sizes_map_to_hm.append(obj_size_map_to_hm)
            offsets_hm.append(offset_hm)
        print()
        
        # to ndarray
        heatmaps = np.array(heatmaps)
        obj_sizes_map_to_hm = np.array(obj_sizes_map_to_hm)
        offsets_hm = np.array(offsets_hm)
        
        return heatmaps, obj_sizes_map_to_hm, offsets_hm

    def calc_heatmap_size_offset_one_data(self, upleft_points, object_sizes):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes ), shape=(num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes ), shape=(num_classes, num_keypoint, 2)
        
        return
            def SHAPE(N) = (hm_y, hm_x, N)

            heatmaps : shape = SHAPE(num_classes), [[[hm], ...], ...]
            object sizes (width,height) in image scale : shape = SHAPE(2 * num_classes),
                                                         [[[w, h, w, h, ...], ...], ...]
            offsets (x,y) in heatmap scale : shape = SHAPE(2 * num_classes),
                                             [[[ofs_x, ofs_y, ofs_x, ofs_y, ...], ...], ...]
        """
        heatmaps = []
        obj_sizes_map_to_hm = []
        offsets_hm = []
        # loop of class
        for iclass in range(self.NUM_CLASSES):
            uplf_pt = upleft_points[iclass]
            obj_size = object_sizes[iclass]

            is_including_keypoint = len(uplf_pt) != 0

            # calc heatmap, object size, offset of one class
            if is_including_keypoint:
                # calculate center point
                # center point without discretization in heat map, shape=(num_keypoint, 2)
                center_point_hm_wo_discretized = (uplf_pt + obj_size * 0.5) * self.to_heatmap_scale
                # center point in heat map, shape=(num_keypoint, 2)
                center_point_hm = np.floor(center_point_hm_wo_discretized)
                center_point_hm = center_point_hm.astype('int')

                # calculate heatmap
                # heatmap, shape=(hm_y, hm_x)
                heatmap = self.__calc_heatmap_one_class(center_point_hm, obj_size)
                # add shape(hm_y, hm_x, 1)
                heatmaps.append(heatmap[:,:,np.newaxis])

                # object size, shape=(hm_y, hm_x, 2)
                obj_size_map_to_hm = self.__calc_object_size_one_class(center_point_hm, obj_size)
                obj_sizes_map_to_hm.append(obj_size_map_to_hm)

                # offset in heatmap, shape=(hm_y, hm_x, 2)
                offset = self.__calc_offset_one_class(center_point_hm_wo_discretized, center_point_hm)
                offsets_hm.append(offset)
            else:
                # heatmap, shape=(hm_y, hm_x)
                heatmap = np.zeros(self.heatmap_size)
                # add shape(hm_y, hm_x, 1)
                heatmaps.append(heatmap[:,:,np.newaxis])

                # object size, shape=(hm_y, hm_x, 2)
                obj_size_hm = np.zeros(self.heatmap_size + (2,))
                obj_sizes_map_to_hm.append(obj_size_hm)

                # offset in heatmap, shape=(hm_y, hm_x, 2)
                offset = np.zeros(self.heatmap_size + (2,))
                offsets_hm.append(offset)

        # list [ndarr1, ndarr2, ...]  ->  concatenate([ndarr1, ndarr2, ...])
        # [shape=(hm_y, hm_x, 1), shape=(hm_y, hm_x, 1), ...] -> shape=(hm_y, hm_x, num_class)
        heatmaps = np.concatenate(heatmaps, axis=-1)
        # [shape=(hm_y, hm_x, 2), shape=(hm_y, hm_x, 2), ...] -> shape=(hm_y, hm_x, 2 * num_class)
        obj_sizes_map_to_hm = np.concatenate(obj_sizes_map_to_hm, axis=-1)
        # [shape=(hm_y, hm_x, 2), shape=(hm_y, hm_x, 2), ...] -> shape=(hm_y, hm_x, 2 * num_class)
        offsets_hm = np.concatenate(offsets_hm, axis=-1)

        return heatmaps, obj_sizes_map_to_hm, offsets_hm

    def __calc_heatmap_one_class(self, center_point_hm, obj_size):
        """
        calculate heatmap of one class.

        center_point_hm: center point in heatmap. shape=(num_keypoint, 2)
        obj_size: object size in image. shape=(num_keypoint, 2)

        """
        # heatmap, shape=(hm_y, hm_x)
        heatmap = np.zeros(self.heatmap_size)
        obj_size_hm = obj_size * self.to_heatmap_scale

        # loop of center point
        x_points_hm = np.arange(self.heatmap_size[1])
        y_points_hm = np.arange(self.heatmap_size[0])
        for cp, obj_sz in zip(center_point_hm, obj_size_hm):
            temp_hm_x = np.exp(-(((x_points_hm - cp[0]) / (obj_sz[0]/2/3)) ** 2) / 2)
            temp_hm_y = np.exp(-(((y_points_hm - cp[1]) / (obj_sz[1]/2/3)) ** 2) / 2)
            temp_hm = temp_hm_x.reshape(1, -1) * temp_hm_y.reshape(-1, 1)
            # maximum
            heatmap = np.maximum(heatmap, temp_hm)

        return heatmap

    def __calc_object_size_one_class(self, center_point_hm, obj_size):
        """
        calculate object size of one class.
        DO NOT scale to heatmap.

        center_point_hm: center point in heatmap. shape=(num_keypoint, 2)
        obj_size: object size in image. shape=(num_keypoint, 2)

        """
        obj_size_map_to_hm = np.zeros(self.heatmap_size + (2,))
        for cp, obj_sz in zip(center_point_hm, obj_size):
            obj_size_map_to_hm[cp[1], cp[0]] = obj_sz

        return obj_size_map_to_hm

    def __calc_offset_one_class(self, center_point_hm_wo_discretized, center_point_hm):
        """
        center_point_hm_wo_discretized: center point in heatmap. shape=(num_keypoint, 2)
        center_point_hm: center point in heatmap. shape=(num_keypoint, 2)
        """
        offset = np.zeros(self.heatmap_size + (2,))
        for cp_wo_dicrt, cp in zip(center_point_hm_wo_discretized, center_point_hm):
            offset[cp[1], cp[0]] = cp_wo_dicrt - cp

        return offset

class ConvertCenterNetOutput:
    def __init__(self, num_classes, image_shape):
        """
        image_shape : rgb=(y,x,3), gray=(y,x,1)
        """
        self.NUM_CLASSES = num_classes
        self.IMAGE_SHAPE = image_shape
        
        # heat map size
        self.heatmap_size = self.__calc_heatmap_size(self.IMAGE_SHAPE[0:2])
        self.to_heatmap_scale = np.array([self.heatmap_size[0] / self.IMAGE_SHAPE[0], self.heatmap_size[1] / self.IMAGE_SHAPE[1]])

        return

    def __calc_heatmap_size(self, input_size):
        """
        input becomes 1/4 in two (2,2) stride operation.
        """
        ndarr_input_size = np.array(input_size, dtype='int')
        hm_size = (ndarr_input_size + ndarr_input_size % 2) // 2
        hm_size = (hm_size + hm_size % 2) // 2
        return tuple(hm_size.tolist())

    def to_upleft_points_object_sizes(self, pred_heatmaps, pred_obj_sizes, pred_offsets, iou_threshold, score_threshold):
        """
        def SHAPE(N) = (num_data, hm_y, hm_x, N)

                heatmaps : shape = SHAPE(num_classes), [[[[hm], ...], ...], ...]
                object sizes (width,height) in image scale : shape = SHAPE(2 * num_classes),
                                                                [[[[w, h, w, h, ...], ...], ...], ...]
                offsets (x,y) in heatmap scale : shape = SHAPE(2 * num_classes),
                                                    [[[[ofs_x, ofs_y, ofs_x, ofs_y, ...], ...], ...], ...]
        """
        conved_upleft_points = []
        conved_object_sizes = []

        # loop of data
        for hm, obj_size, offset in zip(pred_heatmaps, pred_obj_sizes, pred_offsets):
            
            selected_uplefts = []
            selected_objsizes = []
            # loop of class
            for iclass in range(self.NUM_CLASSES):
                # get heatmap in class
                start = iclass
                hm_cls = hm[:,:, start:start+1]
                # score, shape=(hm_x*hm_y)
                score = hm_cls.flatten()

                # get offset
                start = 2 * iclass
                offset_cls = offset[:,:, start:start+2]

                # center point (x, y)
                cx, cy = np.meshgrid(np.arange(self.heatmap_size[1]), np.arange(self.heatmap_size[0]))
                # heat map point
                center_point = np.concatenate((cx[:,:,np.newaxis], cy[:,:,np.newaxis]), axis=-1)
                # correction by offset
                center_point = center_point + offset
                # rescale to image
                center_point = center_point / np.array(self.to_heatmap_scale)[::-1]
                # to shape=(hm_x*hm_y, 2)
                center_point = np.reshape(center_point.flatten(), (-1, 2))

                # get object size in class
                start = 2 * iclass
                obj_size_cls = obj_size[:,:, start:start+2]
                # width, hight
                wh = np.reshape(obj_size.flatten(), (-1, 2))

                # box, concatenate(upper left, bottom right)
                box = np.concatenate((center_point - wh / 2, center_point + wh / 2), axis=-1)

                # selected object point by non maximum suppression
                selected_box = op_util.nms(boxes=box, scores=score, 
                                        iou_threshold=iou_threshold, score_threshold=score_threshold)
                selected_upleft = selected_box[:,0:2]
                selected_objsize = selected_box[:,2:4] - selected_box[:,0:2]

                selected_uplefts.append(selected_upleft)
                selected_objsizes.append(selected_objsize)

            conved_upleft_points.append(np.array(selected_uplefts))
            conved_object_sizes.append(np.array(selected_objsizes))
        
        return conved_upleft_points, conved_object_sizes

    def to_upleft_points_object_sizes_wo_offset(self, pred_heatmaps, pred_obj_sizes, iou_threshold, score_threshold):
        """
        def SHAPE(N) = (num_data, hm_y, hm_x, N)

                heatmaps : shape = SHAPE(num_classes), [[[[hm], ...], ...], ...]
                object sizes (width,height) in image scale : shape = SHAPE(2 * num_classes),
                                                                [[[[w, h, w, h, ...], ...], ...], ...]
                offsets (x,y) in heatmap scale : shape = SHAPE(2 * num_classes),
                                                    [[[[ofs_x, ofs_y, ofs_x, ofs_y, ...], ...], ...], ...]
        """
        conved_upleft_points = []
        conved_object_sizes = []

        # loop of data
        #for hm, obj_size, offset in zip(pred_heatmaps, pred_obj_sizes, pred_offsets):
        for hm, obj_size in zip(pred_heatmaps, pred_obj_sizes):
            
            selected_uplefts = []
            selected_objsizes = []
            # loop of class
            for iclass in range(self.NUM_CLASSES):
                # get heatmap in class
                start = iclass
                hm_cls = hm[:,:, start:start+1]
                # score, shape=(hm_x*hm_y)
                score = hm_cls.flatten()

                # get offset
                #start = 2 * iclass
                #offset_cls = offset[:,:, start:start+2]

                # center point (x, y)
                cx, cy = np.meshgrid(np.arange(self.heatmap_size[1]), np.arange(self.heatmap_size[0]))
                # heat map point
                center_point = np.concatenate((cx[:,:,np.newaxis], cy[:,:,np.newaxis]), axis=-1)
                # correction by offset
                #center_point = center_point + offset
                # rescale to image
                center_point = center_point / np.array(self.to_heatmap_scale)[::-1]
                # to shape=(hm_x*hm_y, 2)
                center_point = np.reshape(center_point.flatten(), (-1, 2))

                # get object size in class
                start = 2 * iclass
                obj_size_cls = obj_size[:,:, start:start+2]
                # width, hight
                wh = np.reshape(obj_size.flatten(), (-1, 2))

                # box, concatenate(upper left, bottom right)
                box = np.concatenate((center_point - wh / 2, center_point + wh / 2), axis=-1)

                # selected object point by non maximum suppression
                selected_box = op_util.nms(boxes=box, scores=score, 
                                        iou_threshold=iou_threshold, score_threshold=score_threshold)
                selected_upleft = selected_box[:,0:2]
                selected_objsize = selected_box[:,2:4] - selected_box[:,0:2]

                selected_uplefts.append(selected_upleft)
                selected_objsizes.append(selected_objsize)

            conved_upleft_points.append(np.array(selected_uplefts))
            conved_object_sizes.append(np.array(selected_objsizes))
        
        return conved_upleft_points, conved_object_sizes
