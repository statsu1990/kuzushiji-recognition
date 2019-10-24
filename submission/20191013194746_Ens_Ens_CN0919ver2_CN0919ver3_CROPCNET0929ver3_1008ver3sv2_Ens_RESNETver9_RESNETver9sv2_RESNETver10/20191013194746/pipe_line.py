import os
import copy
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics

from kuzushiji_data import visualization as visu
from kuzushiji_data import kuzushiji_data_proc as kzsj_data
from kuzushiji_data import evaluation as kzsj_eval

from image_processing import image_proc
from bounding_box_processing import bound_box_proc as bbox_proc

from char_detection.object_point import centernet, data_operator as cn_data_op , util as op_util
from char_detection.object_point import tta as detec_tta
from char_classification import resnet
from char_classification import tta as classification_tta


class CenternetPipeline_190909Ver2():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'test190909_ver2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        conv_images = self.__conv_data_to_input(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

    def __iou_score(self, true_upleft_points, true_object_sizes, pred_upleft_points, pred_object_sizes):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])

        return to_ave(presicion_ious), to_ave(recall_ious), to_ave(f_ious)

class CenternetPipeline_19091701Ver1():
    """
    image preprocessing: linear gamma correction, ben's processing
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'test190909_ver2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = os.path.join('.', 'result', 'CenternetPipeline_19091701', 'centernet')
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = os.path.join(save_dir, 'trained_model.h5')
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        start = 0
        end = 3
        visu.Visualization.visualize_gray_img(tr_imgs[0] * 127.5 + 127.5)
        self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_19091801Ver2():
    """
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CenternetPipeline_19091801_ver2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def run_val_eval(self):
        SEED = 2020

        # use indexs
        #tr_data_idxes = self.__use_indexes(use_train_data=True, 
        #                                   use_val_data=False, 
        #                                   seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        #tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        #tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        #self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
        #             val_imgs, val_upleft_points, val_object_sizes)

        # load
        self.load_model()

        # predict
        #pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        #print('\ntrain data')
        #pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
        #                                                        pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_19091802Ver3():
    """
    image preprocessing: linear gamma correction, ben's processing(base=255)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CenternetPipeline_19091802_ver3', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=255)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_19092201Ver2_tta():
    """
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CenternetPipeline_19091801_ver2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')

        self.TTA_RATIO_SHIFT_TO_GAP_W = 0.2
        self.TTA_RATIO_SHIFT_TO_GAP_H = 0.2

        self.TTA_IOU_THRESHOLD = self.IOU_THRESHOLD
        self.TTA_SCORE_THRESHOLD = np.minimum(0.5, self.SCORE_THRESHOLD)
        self.TTA_SCORE_THRESHOLD_FOR_WEIGHT = 0.5

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def run_val_eval(self):
        SEED = 2020

        # use indexs
        #tr_data_idxes = self.__use_indexes(use_train_data=True, 
        #                                   use_val_data=False, 
        #                                   seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        #tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        #tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        #self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
        #             val_imgs, val_upleft_points, val_object_sizes)

        # load
        self.load_model()

        # predict
        #pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        #print('\ntrain data')
        #pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
        #                                                        pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        start = 0
        end = 20
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)

        # augmentation class instance
        tta = detec_tta.TranslateAugmentation_9case_ThreshScoreWeightedAve(
                                        ratio_shift_to_gap_w=self.TTA_RATIO_SHIFT_TO_GAP_W, 
                                        ratio_shift_to_gap_h=self.TTA_RATIO_SHIFT_TO_GAP_H,
                                        iou_threshold=self.TTA_IOU_THRESHOLD, 
                                        score_threshold=self.TTA_SCORE_THRESHOLD,
                                        score_threshold_for_weight=self.TTA_SCORE_THRESHOLD_FOR_WEIGHT)

        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict_tta(conv_images, tta)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.cnet1 = CenternetPipeline_19091801Ver2(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.cnet2 = CenternetPipeline_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet1.load_model()
        self.cnet2.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.cnet1.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.cnet1.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        pred_heatmaps1, pred_obj_sizes1, pred_offsets1 = self.cnet1.predict(raw_images, need_conv_img=need_conv_img)
        pred_heatmaps2, pred_obj_sizes2, pred_offsets2 = self.cnet2.predict(raw_images, need_conv_img=need_conv_img)

        pred_heatmaps = (pred_heatmaps1 + pred_heatmaps2) * 0.5
        pred_obj_sizes = (pred_obj_sizes1 + pred_obj_sizes2) * 0.5
        pred_offsets = (pred_offsets1 + pred_offsets2) * 0.5

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_19092401Ver2sv2():
    """
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CenternetPipeline_19092401_ver2sv2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.0005
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def run_val_eval(self):
        SEED = 2020

        # use indexs
        #tr_data_idxes = self.__use_indexes(use_train_data=True, 
        #                                   use_val_data=False, 
        #                                   seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        #tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        #tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        #self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
        #             val_imgs, val_upleft_points, val_object_sizes)

        # load
        self.load_model()

        # predict
        #pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        #print('\ntrain data')
        #pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
        #                                                        pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CenternetPipeline_test():
    """
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CenternetPipeline_test', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_conv_img(self, indexes, use_train_data, return_img_size=False, need_print=False):
        imgs = []
        sizes = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[2], img.shape[1]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img[0])
        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes
        else:
            return imgs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.0005
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        tr_data_idxes = np.arange(5)
        val_data_idxes = np.arange(5, 10)

        # data
        # image
        tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        
        for i in range(5):
            self.__visualize(tr_imgs[i:i+1] * 127.5 + 127.5, tr_upleft_points[i:i+1], tr_object_sizes[i:i+1])
            
            outmost_posi = bbox_proc.BoundingBoxProcessing.outermost_position(tr_upleft_points[i], tr_object_sizes[i])
            
            upleft_xs = outmost_posi[0]
            upleft_ys = outmost_posi[1]
            widths = outmost_posi[2] - outmost_posi[0]
            heights = outmost_posi[3] - outmost_posi[1]
            croped_img = image_proc.ImageProcessing.crop(tr_imgs[i], [upleft_xs], [upleft_ys], [widths], [heights])
            croped_img = croped_img[0]

            croped_uplfs, croped_obj_szs = bbox_proc.BoundingBoxProcessing.crop_image(tr_upleft_points[i], 
                                                                tr_object_sizes[i], 
                                                                [upleft_xs, upleft_ys],
                                                                [widths, heights],
                                                                remove_out_bbox=True)
            visu.Visualization.visualize_pred_result(croped_img * 127.5 + 127.5, croped_uplfs[0], croped_obj_szs[0])

            inv_croped_uplfs, inv_croped_obj_szs = bbox_proc.BoundingBoxProcessing.inv_crop_image(croped_uplfs, 
                                                                croped_obj_szs, 
                                                                [upleft_xs, upleft_ys]
                                                                )
            visu.Visualization.visualize_pred_result(tr_imgs[i] * 127.5 + 127.5, inv_croped_uplfs[0], inv_croped_obj_szs[0])

        



        
        
        
        
        
        
        
        
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        
        return

    def run_val_eval(self):
        SEED = 2020

        # use indexs
        #tr_data_idxes = self.__use_indexes(use_train_data=True, 
        #                                   use_val_data=False, 
        #                                   seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        #tr_imgs, tr_img_sizes = self.__read_and_conv_img(tr_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        val_imgs, val_img_sizes = self.__read_and_conv_img(val_data_idxes, use_train_data=True, return_img_size=True, need_print=True)
        # bbox
        #tr_upleft_points, tr_object_sizes = self.__read_train_upleftpoint_size(tr_data_idxes, image_sizes=tr_img_sizes)
        val_upleft_points, val_object_sizes = self.__read_train_upleftpoint_size(val_data_idxes, image_sizes=val_img_sizes)
        #self.__visualize(tr_imgs[0:3] * 127.5 + 127.5, tr_upleft_points[0:3], tr_object_sizes[0:3])

        # train
        #self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
        #             val_imgs, val_upleft_points, val_object_sizes)

        # load
        self.load_model()

        # predict
        #pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        #print('\ntrain data')
        #pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
        #                                                        pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes


class CropedCenternetPipeline_19092901Ver1():
    """
    for croped image.
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CropedCenternetPipeline_19092901_ver1', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')

        self.CROP_MARGIN_SIZE = (4 * 6) * 4

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_crop_and_conv_img_bbox(self, indexes, return_img_size=False, need_print=False):
        use_train_data = True
        
        imgs = []
        sizes = []
        uplfs = []
        obj_szs = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # read bbox
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata, image_sizes=None)

            #visu.Visualization.visualize_pred_result(img[0], upleft_points[0][0], object_sizes[0][0])

            # crop
            img, upleft_points, object_sizes = self.crop_image_including_all_bbox(img[0], 
                                                                                    upleft_points[0], 
                                                                                    object_sizes[0], 
                                                                                    return_bbox=True)

            #visu.Visualization.visualize_pred_result(img, upleft_points[0], object_sizes[0])

            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[1], img.shape[0]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img)

            # resize bbox
            conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(upleft_points, 
                                                                              object_sizes, 
                                                                              (img.shape[1], img.shape[0]), 
                                                                              self.INPUT_SIZE)
            
            #visu.Visualization.visualize_pred_result(conv_img * 127.5 + 127.5, conv_uplf[0], conv_sz[0])

            uplfs.append(conv_uplf)
            obj_szs.append(conv_sz)

        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes, uplfs, obj_szs
        else:
            return imgs, uplfs, obj_szs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def crop_image_including_all_bbox(self, image, upleft_points, obj_sizes, return_bbox=True):
        """
        Returns:
            croped image: image including all bbox
            upleft_points after croping: return if return_bbox==True
            obj_sizes after croping: return if return_bbox==True
        """
        
        upleft_x, upleft_y, width, height = self.__calc_crop_range([image.shape[1], image.shape[0]], upleft_points, obj_sizes)

        if upleft_x is None:
            """
            have no bbox
            """
            croped_img = copy.copy(image)
            if return_bbox:
                croped_uplfs = copy.copy(upleft_points)
                croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped image
            croped_img = image_proc.ImageProcessing.crop(image, [upleft_x], [upleft_y], [width], [height])
            croped_img = croped_img[0]

            if return_bbox:
                # croped bbox
                croped_uplfs, croped_obj_szs = bbox_proc.BoundingBoxProcessing.crop_image(upleft_points, 
                                                                        obj_sizes, 
                                                                        [upleft_x, upleft_y],
                                                                        [width, height],
                                                                        remove_out_bbox=True)

        if return_bbox:
            return croped_img, croped_uplfs, croped_obj_szs
        else:
            return croped_img

    def inv_crop_image_including_all_bbox(self, image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes):

        upleft_x, upleft_y, width, height = self.__calc_crop_range(image_size_wh_before_crop, 
                                                                     upleft_points_before_crop, 
                                                                     obj_sizes_before_crop)

        if upleft_x is None:
            """
            have no bbox
            """
            inv_croped_uplfs = copy.copy(upleft_points)
            inv_croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped bbox
            inv_croped_uplfs, inv_croped_obj_szs = bbox_proc.BoundingBoxProcessing.inv_crop_image(
                                                                    upleft_points, 
                                                                    obj_sizes, 
                                                                    croped_upleft_point=[upleft_x, upleft_y],
                                                                    )

        return inv_croped_uplfs, inv_croped_obj_szs

    def __calc_crop_range(self, image_size_wh, upleft_points, obj_sizes):
        """
        Returns:
            upleft_x, upleft_y, width, upleft_y
        """
        # outer most position of all bbox
        outermost_posi = bbox_proc.BoundingBoxProcessing.outermost_position(upleft_points, obj_sizes)

        if len(outermost_posi) == 0:
            """
            have no bbox
            """
            return None, None, None, None
        else:
            # temp upleft
            temp_upleft_x = np.maximum(0, outermost_posi[0] - self.CROP_MARGIN_SIZE)
            temp_upleft_y = np.maximum(0, outermost_posi[1] - self.CROP_MARGIN_SIZE)
            temp_bottomright_x = np.minimum(image_size_wh[0], outermost_posi[2] + self.CROP_MARGIN_SIZE)
            temp_bottomright_y = np.minimum(image_size_wh[1], outermost_posi[3] + self.CROP_MARGIN_SIZE)

            # center
            min_w = self.INPUT_SIZE[0]
            center_x = (temp_upleft_x + temp_bottomright_x) * 0.5
            center_x = np.clip(center_x, min_w / 2, image_size_wh[0] - min_w / 2)

            min_h = self.INPUT_SIZE[1]
            center_y = (temp_upleft_y + temp_bottomright_y) * 0.5
            center_y = np.clip(center_y, min_h / 2, image_size_wh[1] - min_h / 2)

            # width
            width = np.maximum(min_w, temp_bottomright_x - temp_upleft_x)
            height = np.maximum(min_h, temp_bottomright_y - temp_upleft_y)

            # up left
            upleft_x = center_x - width * 0.5
            upleft_y = center_y - height * 0.5

            return upleft_x, upleft_y, width, height

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = 0.0 #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = tr_data_idxes[:20]
        #val_data_idxes = val_data_idxes[:20]

        # data
        # image and bbox
        tr_imgs, tr_upleft_points, tr_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                tr_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)
        val_imgs, val_upleft_points, val_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                val_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)

        #for i in tr_data_idxes:
        #    self.__visualize(tr_imgs[i:i+1] * 127.5 + 127.5, tr_upleft_points[i:i+1], tr_object_sizes[i:i+1])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        start = 0
        end = 3
        self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class DetecPipeline_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19092901Ver1():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.cnet = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.croped_cnet = CropedCenternetPipeline_19092901Ver1(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet.load_model()
        self.croped_cnet.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        pred_upleft_points, pred_object_sizes = self.cnet.predict_bbox(raw_images, need_conv_img=need_conv_img)

        def _pred_bbox_one_sample(_img, _uplfs, _obj_szs):
            #visu.Visualization.visualize_pred_result(_img, _uplfs[0], _obj_szs[0])

            _croped_img = self.croped_cnet.crop_image_including_all_bbox(_img, _uplfs, _obj_szs, return_bbox=False)
            _pred_uplfs, _pred_obj_szs = self.croped_cnet.predict_bbox(_croped_img[np.newaxis], need_conv_img=need_conv_img)

            #visu.Visualization.visualize_pred_result(_croped_img, _pred_uplfs[0][0], _pred_obj_szs[0][0])

            _pred_uplfs, _pred_obj_szs = self.croped_cnet.inv_crop_image_including_all_bbox(_img.shape[1::-1], 
                                                                                            _uplfs, _obj_szs, 
                                                                                            _pred_uplfs[0], _pred_obj_szs[0])

            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            return _pred_uplfs, _pred_obj_szs

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = _pred_bbox_one_sample(raw_images, pred_upleft_points[0], pred_object_sizes[0])
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for imgs, p_uplf, p_obj_sz in zip(raw_images, pred_upleft_points, pred_object_sizes):
                uplfs, obj_szs = _pred_bbox_one_sample(imgs, p_uplf, p_obj_sz)
                uplf_points.append(uplfs)
                obj_sizes.append(obj_szs)

        #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

        return uplf_points, obj_sizes

class DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19092901Ver1():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.cnet = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.croped_cnet = CropedCenternetPipeline_19092901Ver1(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet.load_model()
        self.croped_cnet.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        pred_upleft_points, pred_object_sizes = self.cnet.predict_bbox(raw_images, need_conv_img=need_conv_img)

        def _pred_bbox_one_sample(_img, _uplfs, _obj_szs):
            #visu.Visualization.visualize_pred_result(_img, _uplfs[0], _obj_szs[0])

            _croped_img = self.croped_cnet.crop_image_including_all_bbox(_img, _uplfs, _obj_szs, return_bbox=False)
            _pred_uplfs, _pred_obj_szs = self.croped_cnet.predict_bbox(_croped_img[np.newaxis], need_conv_img=need_conv_img)
            #visu.Visualization.visualize_pred_result(_croped_img, _pred_uplfs[0][0], _pred_obj_szs[0][0])

            _pred_uplfs, _pred_obj_szs = self.croped_cnet.inv_crop_image_including_all_bbox(_img.shape[1::-1], 
                                                                                            _uplfs, _obj_szs, 
                                                                                            _pred_uplfs[0], _pred_obj_szs[0])
            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            # have no bbox
            if _uplfs.shape[1] != 0:
                _pred_uplfs, _pred_obj_szs = self.nms_bbox([_uplfs[0], _pred_uplfs[0]], 
                                                           [_obj_szs[0], _pred_obj_szs[0]], 
                                                           self.IOU_THRESHOLD)
                _pred_uplfs, _pred_obj_szs = _pred_uplfs[np.newaxis], _pred_obj_szs[np.newaxis]

            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            return _pred_uplfs, _pred_obj_szs

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = _pred_bbox_one_sample(raw_images, pred_upleft_points[0], pred_object_sizes[0])
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for imgs, p_uplf, p_obj_sz in zip(raw_images, pred_upleft_points, pred_object_sizes):
                uplfs, obj_szs = _pred_bbox_one_sample(imgs, p_uplf, p_obj_sz)
                uplf_points.append(uplfs)
                obj_sizes.append(obj_szs)

        #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

        return uplf_points, obj_sizes

    def nms_bbox(self, upleft_points_list, object_sizes_list, iou_threshold):
        """
        Args:
            upleft_points_list: [upleft_points1, upleft_points2, ...], upleft_points shape = (num_box, 2)
            object_sizes_list: [object_sizes1, object_sizes2, ...], object_sizes shape = (num_box, 2)
        """
        uplfs = np.concatenate(upleft_points_list, axis=0)
        obj_szs = np.concatenate(object_sizes_list, axis=0)
        
        bottomrights = uplfs + obj_szs
        boxes = np.concatenate([uplfs, bottomrights], axis=1)

        scores = obj_szs[:,0] * obj_szs[:,1]

        nms_boxes = op_util.nms(boxes, scores, iou_threshold)
        nms_uplfs = nms_boxes[:,:2]
        nms_obj_szs = nms_boxes[:,2:4] - nms_boxes[:,:2]

        return nms_uplfs, nms_obj_szs

class CropedCenternetPipeline_19100101Ver2():
    """
    for croped image.
    image preprocessing: linear gamma correction, ben's processing(base=128)
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CropedCenternetPipeline_19100101_ver2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')

        self.CROP_MARGIN_SIZE = 4

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_crop_and_conv_img_bbox(self, indexes, return_img_size=False, need_print=False):
        use_train_data = True
        
        imgs = []
        sizes = []
        uplfs = []
        obj_szs = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # read bbox
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata, image_sizes=None)

            #visu.Visualization.visualize_pred_result(img[0], upleft_points[0][0], object_sizes[0][0])

            # crop
            img, upleft_points, object_sizes = self.crop_image_including_all_bbox(img[0], 
                                                                                    upleft_points[0], 
                                                                                    object_sizes[0], 
                                                                                    return_bbox=True)

            #visu.Visualization.visualize_pred_result(img, upleft_points[0], object_sizes[0])

            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[1], img.shape[0]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img)

            # resize bbox
            conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(upleft_points, 
                                                                              object_sizes, 
                                                                              (img.shape[1], img.shape[0]), 
                                                                              self.INPUT_SIZE)
            
            #visu.Visualization.visualize_pred_result(conv_img * 127.5 + 127.5, conv_uplf[0], conv_sz[0])

            uplfs.append(conv_uplf)
            obj_szs.append(conv_sz)

        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes, uplfs, obj_szs
        else:
            return imgs, uplfs, obj_szs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def crop_image_including_all_bbox(self, image, upleft_points, obj_sizes, return_bbox=True):
        """
        Returns:
            croped image: image including all bbox
            upleft_points after croping: return if return_bbox==True
            obj_sizes after croping: return if return_bbox==True
        """
        
        upleft_x, upleft_y, width, height = self.__calc_crop_range([image.shape[1], image.shape[0]], upleft_points, obj_sizes)

        if upleft_x is None:
            """
            have no bbox
            """
            croped_img = copy.copy(image)
            if return_bbox:
                croped_uplfs = copy.copy(upleft_points)
                croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped image
            croped_img = image_proc.ImageProcessing.crop(image, [upleft_x], [upleft_y], [width], [height])
            croped_img = croped_img[0]

            if return_bbox:
                # croped bbox
                croped_uplfs, croped_obj_szs = bbox_proc.BoundingBoxProcessing.crop_image(upleft_points, 
                                                                        obj_sizes, 
                                                                        [upleft_x, upleft_y],
                                                                        [width, height],
                                                                        remove_out_bbox=True)

        if return_bbox:
            return croped_img, croped_uplfs, croped_obj_szs
        else:
            return croped_img

    def inv_crop_image_including_all_bbox(self, image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes):

        upleft_x, upleft_y, width, height = self.__calc_crop_range(image_size_wh_before_crop, 
                                                                     upleft_points_before_crop, 
                                                                     obj_sizes_before_crop)

        if upleft_x is None:
            """
            have no bbox
            """
            inv_croped_uplfs = copy.copy(upleft_points)
            inv_croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped bbox
            inv_croped_uplfs, inv_croped_obj_szs = bbox_proc.BoundingBoxProcessing.inv_crop_image(
                                                                    upleft_points, 
                                                                    obj_sizes, 
                                                                    croped_upleft_point=[upleft_x, upleft_y],
                                                                    )

        return inv_croped_uplfs, inv_croped_obj_szs

    def __calc_crop_range(self, image_size_wh, upleft_points, obj_sizes):
        """
        Returns:
            upleft_x, upleft_y, width, upleft_y
        """
        # outer most position of all bbox
        outermost_posi = bbox_proc.BoundingBoxProcessing.outermost_position(upleft_points, obj_sizes)

        if len(outermost_posi) == 0:
            """
            have no bbox
            """
            return None, None, None, None
        else:
            # temp upleft
            temp_upleft_x = np.maximum(0, outermost_posi[0] - self.CROP_MARGIN_SIZE)
            temp_upleft_y = np.maximum(0, outermost_posi[1] - self.CROP_MARGIN_SIZE)
            temp_bottomright_x = np.minimum(image_size_wh[0], outermost_posi[2] + self.CROP_MARGIN_SIZE)
            temp_bottomright_y = np.minimum(image_size_wh[1], outermost_posi[3] + self.CROP_MARGIN_SIZE)

            # center
            min_w = self.INPUT_SIZE[0]
            center_x = (temp_upleft_x + temp_bottomright_x) * 0.5
            center_x = np.clip(center_x, min_w / 2, image_size_wh[0] - min_w / 2)

            min_h = self.INPUT_SIZE[1]
            center_y = (temp_upleft_y + temp_bottomright_y) * 0.5
            center_y = np.clip(center_y, min_h / 2, image_size_wh[1] - min_h / 2)

            # width
            width = np.maximum(min_w, temp_bottomright_x - temp_upleft_x)
            height = np.maximum(min_h, temp_bottomright_y - temp_upleft_y)

            # up left
            upleft_x = center_x - width * 0.5
            upleft_y = center_y - height * 0.5

            return upleft_x, upleft_y, width, height

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = 0.1 #0.1
        ZOOM_RATE = None #0.1
        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None)
        
        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []

        true_detec_num = 0
        pred_detec_num = 0
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

            true_detec_num = true_detec_num + true_uplf.size
            pred_detec_num = pred_detec_num + pred_uplf.size

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
            print('true_detec_num, pred_detec_num, pred/true : {0}, {1}, {2:.3f}'.format(true_detec_num, pred_detec_num, pred_detec_num/true_detec_num))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.random.choice(tr_data_idxes, 5, False)
        #val_data_idxes = np.random.choice(val_data_idxes, 5, False)

        # data
        # image and bbox
        tr_imgs, tr_upleft_points, tr_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                tr_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)
        val_imgs, val_upleft_points, val_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                val_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)

        #for i in tr_data_idxes:
        #    self.__visualize(tr_imgs[i:i+1] * 127.5 + 127.5, tr_upleft_points[i:i+1], tr_object_sizes[i:i+1])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        start = 0
        end = 20
        self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CropedCenternetPipeline_19100201Ver3():
    """
    for croped image.
    image preprocessing: linear gamma correction, ben's processing(base=128)
    data augmentation: shift, random erasing
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CropedCenternetPipeline_19100201_ver3', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')

        self.CROP_MARGIN_SIZE = (4 * 6) * 4

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_crop_and_conv_img_bbox(self, indexes, return_img_size=False, need_print=False):
        use_train_data = True
        
        imgs = []
        sizes = []
        uplfs = []
        obj_szs = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # read bbox
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata, image_sizes=None)

            #visu.Visualization.visualize_pred_result(img[0], upleft_points[0][0], object_sizes[0][0])

            # crop
            img, upleft_points, object_sizes = self.crop_image_including_all_bbox(img[0], 
                                                                                    upleft_points[0], 
                                                                                    object_sizes[0], 
                                                                                    return_bbox=True)

            #visu.Visualization.visualize_pred_result(img, upleft_points[0], object_sizes[0])

            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[1], img.shape[0]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img)

            # resize bbox
            conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(upleft_points, 
                                                                              object_sizes, 
                                                                              (img.shape[1], img.shape[0]), 
                                                                              self.INPUT_SIZE)
            
            #visu.Visualization.visualize_pred_result(conv_img * 127.5 + 127.5, conv_uplf[0], conv_sz[0])

            uplfs.append(conv_uplf)
            obj_szs.append(conv_sz)

        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes, uplfs, obj_szs
        else:
            return imgs, uplfs, obj_szs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def crop_image_including_all_bbox(self, image, upleft_points, obj_sizes, return_bbox=True):
        """
        Returns:
            croped image: image including all bbox
            upleft_points after croping: return if return_bbox==True
            obj_sizes after croping: return if return_bbox==True
        """
        
        upleft_x, upleft_y, width, height = self.__calc_crop_range([image.shape[1], image.shape[0]], upleft_points, obj_sizes)

        if upleft_x is None:
            """
            have no bbox
            """
            croped_img = copy.copy(image)
            if return_bbox:
                croped_uplfs = copy.copy(upleft_points)
                croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped image
            croped_img = image_proc.ImageProcessing.crop(image, [upleft_x], [upleft_y], [width], [height])
            croped_img = croped_img[0]

            if return_bbox:
                # croped bbox
                croped_uplfs, croped_obj_szs = bbox_proc.BoundingBoxProcessing.crop_image(upleft_points, 
                                                                        obj_sizes, 
                                                                        [upleft_x, upleft_y],
                                                                        [width, height],
                                                                        remove_out_bbox=True)

        if return_bbox:
            return croped_img, croped_uplfs, croped_obj_szs
        else:
            return croped_img

    def inv_crop_image_including_all_bbox(self, image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes):

        upleft_x, upleft_y, width, height = self.__calc_crop_range(image_size_wh_before_crop, 
                                                                     upleft_points_before_crop, 
                                                                     obj_sizes_before_crop)

        if upleft_x is None:
            """
            have no bbox
            """
            inv_croped_uplfs = copy.copy(upleft_points)
            inv_croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped bbox
            inv_croped_uplfs, inv_croped_obj_szs = bbox_proc.BoundingBoxProcessing.inv_crop_image(
                                                                    upleft_points, 
                                                                    obj_sizes, 
                                                                    croped_upleft_point=[upleft_x, upleft_y],
                                                                    )

        return inv_croped_uplfs, inv_croped_obj_szs

    def __calc_crop_range(self, image_size_wh, upleft_points, obj_sizes):
        """
        Returns:
            upleft_x, upleft_y, width, upleft_y
        """
        # outer most position of all bbox
        outermost_posi = bbox_proc.BoundingBoxProcessing.outermost_position(upleft_points, obj_sizes)

        if len(outermost_posi) == 0:
            """
            have no bbox
            """
            return None, None, None, None
        else:
            # temp upleft
            temp_upleft_x = np.maximum(0, outermost_posi[0] - self.CROP_MARGIN_SIZE)
            temp_upleft_y = np.maximum(0, outermost_posi[1] - self.CROP_MARGIN_SIZE)
            temp_bottomright_x = np.minimum(image_size_wh[0], outermost_posi[2] + self.CROP_MARGIN_SIZE)
            temp_bottomright_y = np.minimum(image_size_wh[1], outermost_posi[3] + self.CROP_MARGIN_SIZE)

            # center
            min_w = self.INPUT_SIZE[0]
            center_x = (temp_upleft_x + temp_bottomright_x) * 0.5
            center_x = np.clip(center_x, min_w / 2, image_size_wh[0] - min_w / 2)

            min_h = self.INPUT_SIZE[1]
            center_y = (temp_upleft_y + temp_bottomright_y) * 0.5
            center_y = np.clip(center_y, min_h / 2, image_size_wh[1] - min_h / 2)

            # width
            width = np.maximum(min_w, temp_bottomright_x - temp_upleft_x)
            height = np.maximum(min_h, temp_bottomright_y - temp_upleft_y)

            # up left
            upleft_x = center_x - width * 0.5
            upleft_y = center_y - height * 0.5

            return upleft_x, upleft_y, width, height

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = None #0.1
        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5,
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE,
                                                        random_erasing_kwargs=RANDOM_ERASING_KWARGS)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None,
                                                        random_erasing_kwargs=None)
        
        #cnet_datagen[0]

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = tr_data_idxes[:5]
        #val_data_idxes = val_data_idxes[:5]

        # data
        # image and bbox
        tr_imgs, tr_upleft_points, tr_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                tr_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)
        val_imgs, val_upleft_points, val_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                val_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)

        #for i in tr_data_idxes:
        #    self.__visualize(tr_imgs[i:i+1] * 127.5 + 127.5, tr_upleft_points[i:i+1], tr_object_sizes[i:i+1])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CropedCenternetPipeline_19100801Ver3sv2():
    """
    for croped image.
    image preprocessing: linear gamma correction, ben's processing(base=128)
    data augmentation: shift, random erasing
    """
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.__config(iou_thresh, score_thresh)
        self.__initilize()

        return

    def __config(self, iou_thresh=None, score_thresh=None):
        self.INPUT_SIZE = (512, 512)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)

        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5

        self.NUM_STACKS = 1
        self.NUM_CHANNELS = 128
        self.NUM_CHANNEL_HG = [64, 64, 96, 128, 128]
        self.LOSS_COEFS = [1, 0.1, 1]
        self.HM_LOSS_ALPHA = 2
        self.HM_LOSS_BETA = 4

        self.MODEL_DIR = os.path.join('.', 'result', 'CropedCenternetPipeline_19100801_ver3sv2', 'centernet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')

        self.CROP_MARGIN_SIZE = (4 * 6) * 4

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5
            return _conv_img

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            for img in raw_images:
                conv_imgs.append(conv_func_one_sample(img))
            conv_imgs = np.array(conv_imgs)

        return conv_imgs

    def __build_model_instance(self):
        self.cnet = centernet.CenterNet_SHGN(num_classes=1, 
                                        image_shape=self.INPUT_SHAPE, 
                                        num_stacks=self.NUM_STACKS, 
                                        num_channels=self.NUM_CHANNELS, 
                                        num_channel_hg=self.NUM_CHANNEL_HG, 
                                        loss_coefs=self.LOSS_COEFS, 
                                        hm_loss_alpha=self.HM_LOSS_ALPHA, 
                                        hm_loss_beta=self.HM_LOSS_BETA)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_and_crop_and_conv_img_bbox(self, indexes, return_img_size=False, need_print=False):
        use_train_data = True
        
        imgs = []
        sizes = []
        uplfs = []
        obj_szs = []
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=use_train_data, need_print=False)
            
            # read bbox
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata, image_sizes=None)

            #visu.Visualization.visualize_pred_result(img[0], upleft_points[0][0], object_sizes[0][0])

            # crop
            img, upleft_points, object_sizes = self.crop_image_including_all_bbox(img[0], 
                                                                                    upleft_points[0], 
                                                                                    object_sizes[0], 
                                                                                    return_bbox=True)

            #visu.Visualization.visualize_pred_result(img, upleft_points[0], object_sizes[0])

            # size (w, h)
            if return_img_size:
                sizes.append((img.shape[1], img.shape[0]))

            # conv 
            conv_img = self.__conv_data_to_input(img)
            imgs.append(conv_img)

            # resize bbox
            conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(upleft_points, 
                                                                              object_sizes, 
                                                                              (img.shape[1], img.shape[0]), 
                                                                              self.INPUT_SIZE)
            
            #visu.Visualization.visualize_pred_result(conv_img * 127.5 + 127.5, conv_uplf[0], conv_sz[0])

            uplfs.append(conv_uplf)
            obj_szs.append(conv_sz)

        imgs = np.array(imgs)

        if need_print:
            print()

        if return_img_size:
            return imgs, sizes, uplfs, obj_szs
        else:
            return imgs, uplfs, obj_szs

    def __read_train_upleftpoint_size(self, indexes, image_sizes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes=indexes)

        if image_sizes is not None:
            for idata, (uplf, sz, img_sz) in enumerate(zip(upleft_points, object_sizes, image_sizes)):
                conv_uplf, conv_sz = bbox_proc.BoundingBoxProcessing.resize_image(uplf, sz, img_sz, self.INPUT_SIZE)
                upleft_points[idata] = conv_uplf
                object_sizes[idata] = conv_sz

        return upleft_points, object_sizes

    def crop_image_including_all_bbox(self, image, upleft_points, obj_sizes, return_bbox=True):
        """
        Returns:
            croped image: image including all bbox
            upleft_points after croping: return if return_bbox==True
            obj_sizes after croping: return if return_bbox==True
        """
        
        upleft_x, upleft_y, width, height = self.__calc_crop_range([image.shape[1], image.shape[0]], upleft_points, obj_sizes)

        if upleft_x is None:
            """
            have no bbox
            """
            croped_img = copy.copy(image)
            if return_bbox:
                croped_uplfs = copy.copy(upleft_points)
                croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped image
            croped_img = image_proc.ImageProcessing.crop(image, [upleft_x], [upleft_y], [width], [height])
            croped_img = croped_img[0]

            if return_bbox:
                # croped bbox
                croped_uplfs, croped_obj_szs = bbox_proc.BoundingBoxProcessing.crop_image(upleft_points, 
                                                                        obj_sizes, 
                                                                        [upleft_x, upleft_y],
                                                                        [width, height],
                                                                        remove_out_bbox=True)

        if return_bbox:
            return croped_img, croped_uplfs, croped_obj_szs
        else:
            return croped_img

    def inv_crop_image_including_all_bbox(self, image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes):

        upleft_x, upleft_y, width, height = self.__calc_crop_range(image_size_wh_before_crop, 
                                                                     upleft_points_before_crop, 
                                                                     obj_sizes_before_crop)

        if upleft_x is None:
            """
            have no bbox
            """
            inv_croped_uplfs = copy.copy(upleft_points)
            inv_croped_obj_szs = copy.copy(obj_sizes)
        else:
            # croped bbox
            inv_croped_uplfs, inv_croped_obj_szs = bbox_proc.BoundingBoxProcessing.inv_crop_image(
                                                                    upleft_points, 
                                                                    obj_sizes, 
                                                                    croped_upleft_point=[upleft_x, upleft_y],
                                                                    )

        return inv_croped_uplfs, inv_croped_obj_szs

    def __calc_crop_range(self, image_size_wh, upleft_points, obj_sizes):
        """
        Returns:
            upleft_x, upleft_y, width, upleft_y
        """
        # outer most position of all bbox
        outermost_posi = bbox_proc.BoundingBoxProcessing.outermost_position(upleft_points, obj_sizes)

        if len(outermost_posi) == 0:
            """
            have no bbox
            """
            return None, None, None, None
        else:
            # temp upleft
            temp_upleft_x = np.maximum(0, outermost_posi[0] - self.CROP_MARGIN_SIZE)
            temp_upleft_y = np.maximum(0, outermost_posi[1] - self.CROP_MARGIN_SIZE)
            temp_bottomright_x = np.minimum(image_size_wh[0], outermost_posi[2] + self.CROP_MARGIN_SIZE)
            temp_bottomright_y = np.minimum(image_size_wh[1], outermost_posi[3] + self.CROP_MARGIN_SIZE)

            # center
            min_w = self.INPUT_SIZE[0]
            center_x = (temp_upleft_x + temp_bottomright_x) * 0.5
            center_x = np.clip(center_x, min_w / 2, image_size_wh[0] - min_w / 2)

            min_h = self.INPUT_SIZE[1]
            center_y = (temp_upleft_y + temp_bottomright_y) * 0.5
            center_y = np.clip(center_y, min_h / 2, image_size_wh[1] - min_h / 2)

            # width
            width = np.maximum(min_w, temp_bottomright_x - temp_upleft_x)
            height = np.maximum(min_h, temp_bottomright_y - temp_upleft_y)

            # up left
            upleft_x = center_x - width * 0.5
            upleft_y = center_y - height * 0.5

            return upleft_x, upleft_y, width, height

    def __train(self, tr_imgs, tr_upleft_points, tr_object_sizes, 
                val_imgs, val_upleft_points, val_object_sizes):

        # buid model
        self.cnet.build_model()

        # training
        LEARNING_RATE = 0.001
        def LEARNING_RATE_SCHEDULE(_epoch):
            _lr = LEARNING_RATE
            if _epoch >= 320:
                _lr = _lr * 0.01
            elif _epoch >= 240:
                _lr = _lr * 0.05
            elif _epoch >= 160:
                _lr = _lr * 0.1
            return _lr
        
        EPOCHS = 400 #400
        BATCH_SIZE = 16

        # data generator
        SHIFT_BRIGHTNESS = [-0.1, 0.1]
        SHIFT_WIDTH_HEIGHT = True
        CROP_CUT_RATE = None #0.1
        ZOOM_OUT_RATE = None #0.1
        ZOOM_RATE = None #0.1
        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5,
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        cnet_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=tr_imgs, 
                                                        upleft_points=tr_upleft_points, 
                                                        object_sizes=tr_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=SHIFT_BRIGHTNESS, 
                                                        do_shift_width_height=SHIFT_WIDTH_HEIGHT,
                                                        crop_cut_rate=CROP_CUT_RATE,
                                                        zoom_out_rate=ZOOM_OUT_RATE,
                                                        zoom_rate=ZOOM_RATE,
                                                        random_erasing_kwargs=RANDOM_ERASING_KWARGS)
        cnet_val_datagen = cn_data_op.CenterNetDataGenerator(class_num=1, 
                                                        image=val_imgs, 
                                                        upleft_points=val_upleft_points, 
                                                        object_sizes=val_object_sizes, 
                                                        batch_size=BATCH_SIZE,
                                                        shift_brightness_range=None, 
                                                        do_shift_width_height=False,
                                                        crop_cut_rate=None,
                                                        zoom_out_rate=None,
                                                        zoom_rate=None,
                                                        random_erasing_kwargs=None)
        
        #cnet_datagen[0]

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # fit
        self.cnet.train_model_with_generator_vlgene(train_generator=cnet_datagen, 
                                                    val_generator=cnet_val_datagen,
                                                    train_steps_per_epoch=len(tr_imgs)/BATCH_SIZE, 
                                                    epochs=EPOCHS, 
                                                    learning_rate=LEARNING_RATE, 
                                                    lr_sche=LEARNING_RATE_SCHEDULE, 
                                                    save_file=None, 
                                                    csv_file=save_csv_file)
        # save model
        self.cnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __iou_score(self, true_upleft_points, true_object_sizes, 
                    pred_upleft_points, pred_object_sizes, 
                    need_print=False):
        """
        upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
        where num_classes = 1
        """
        def compute_box(_uplfp, _obj_sz):
            _box = np.concatenate((_uplfp, _uplfp + _obj_sz), axis=-1)
            return _box

        presicion_ious = []
        recall_ious = []
        f_ious = []
        # loop of data
        for true_uplf, true_obj_sz, pred_uplf, pred_obj_sz in zip(true_upleft_points, 
                                                                  true_object_sizes, 
                                                                  pred_upleft_points, 
                                                                  pred_object_sizes):
            true_box = compute_box(true_uplf[0], true_obj_sz[0])
            pred_box = compute_box(pred_uplf[0], pred_obj_sz[0])

            presicion_iou, recall_iou, f_iou = op_util.iou_score(true_box, pred_box)
            presicion_ious.append(presicion_iou)
            recall_ious.append(recall_iou)
            f_ious.append(f_iou)

        def to_ave(_list):
            _arr = np.array(_list)
            return np.average(_arr[~np.isnan(_arr)])


        pres_iou = to_ave(presicion_ious)
        recall_iou = to_ave(recall_ious)
        f_iou = to_ave(f_ious)

        if need_print:
            print('pres_iou, recall_iou, f_iou : {0:.3f}, {1:.3f}, {2:.3f}'.format(pres_iou, recall_iou, f_iou))
        
        return pres_iou, recall_iou, f_iou

    def __visualize(self, images, uplf_points, obj_sizes):
        for img, uplf, sz in zip(images, uplf_points, obj_sizes):
            visu.Visualization.visualize_pred_result(img, uplf[0], sz[0])
        return

    def run_train(self):
        SEED = 2021

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = tr_data_idxes[:5]
        #val_data_idxes = val_data_idxes[:5]

        # data
        # image and bbox
        tr_imgs, tr_upleft_points, tr_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                tr_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)
        val_imgs, val_upleft_points, val_object_sizes = self.__read_and_crop_and_conv_img_bbox(
                                                                val_data_idxes, 
                                                                return_img_size=False, 
                                                                need_print=True)

        #for i in tr_data_idxes:
        #    self.__visualize(tr_imgs[i:i+1] * 127.5 + 127.5, tr_upleft_points[i:i+1], tr_object_sizes[i:i+1])

        # train
        self.__train(tr_imgs, tr_upleft_points, tr_object_sizes, 
                     val_imgs, val_upleft_points, val_object_sizes)

        # predict
        pred_tr_uplf_points, pred_tr_obj_sizes = self.predict_bbox(tr_imgs, need_conv_img=False)
        pred_val_uplf_points, pred_val_obj_sizes = self.predict_bbox(val_imgs, need_conv_img=False)

        # iou score
        print('\ntrain data')
        pres_iou_tr, recall_iou_tr, f_iou_tr = self.__iou_score(tr_upleft_points, tr_object_sizes, 
                                                                pred_tr_uplf_points, pred_tr_obj_sizes, need_print=True)
        print('\nval data')
        pres_iou_val, recall_iou_val, f_iou_val = self.__iou_score(val_upleft_points, val_object_sizes, 
                                                                   pred_val_uplf_points, pred_val_obj_sizes, need_print=True)

        # visualize
        #start = 0
        #end = 3
        #self.__visualize(tr_imgs[start:end] * 127.5 + 127.5, pred_tr_uplf_points[start:end], pred_tr_obj_sizes[start:end])
        #self.__visualize(val_imgs[start:end] * 127.5 + 127.5, pred_val_uplf_points[start:end], pred_val_obj_sizes[start:end])

        return

    def load_model(self):
        self.cnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, need_conv_img=True):
        """
        Reeturns:
            pred_heatmaps, pred_obj_sizes, pred_offsets
        """
        if need_conv_img:
            conv_images = self.__conv_data_to_input(raw_images)
        else:
            conv_images = copy.copy(raw_images)
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.cnet.predict(conv_images)
        return pred_heatmaps, pred_obj_sizes, pred_offsets

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        # pred heatmap etc.
        pred_heatmaps, pred_obj_sizes, pred_offsets = self.predict(raw_images, need_conv_img=need_conv_img)

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes

class CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv2():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.cnet1 = CropedCenternetPipeline_19100201Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.cnet2 = CropedCenternetPipeline_19100801Ver3sv2(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def crop_image_including_all_bbox(self, image, upleft_points, obj_sizes, return_bbox=True):
        """
        Returns:
            croped image: image including all bbox
            upleft_points after croping: return if return_bbox==True
            obj_sizes after croping: return if return_bbox==True
        """
        return self.cnet1.crop_image_including_all_bbox(image, upleft_points, obj_sizes, return_bbox)

    def inv_crop_image_including_all_bbox(self, image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes):

        return self.cnet1.inv_crop_image_including_all_bbox(image_size_wh_before_crop, 
                                          upleft_points_before_crop, 
                                          obj_sizes_before_crop, 
                                          upleft_points, 
                                          obj_sizes)


    def load_model(self):
        self.cnet1.load_model()
        self.cnet2.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        # conversion class instance
        conv_cnet_oup = cn_data_op.ConvertCenterNetOutput(num_classes=1, 
                                                          image_shape=self.cnet1.INPUT_SHAPE)
        # func for one sample
        def pred_bbox_one_sample(_hm, _sz, _ofs, _raw_img_sz):
            _uplf_points, _obj_sizes = conv_cnet_oup.to_upleft_points_object_sizes([_hm], 
                                                                                   [_sz], 
                                                                                   [_ofs], 
                                                                                   self.IOU_THRESHOLD, 
                                                                                   self.SCORE_THRESHOLD)
            # shape (num_sample=1, num_class=1, num_obj, 2) - > (num_class=1, num_obj, 2)
            _uplf_points = _uplf_points[0]
            _obj_sizes = _obj_sizes[0]

            # rescale
            if len(_uplf_points) > 0:
                _uplf_points, _obj_sizes = bbox_proc.BoundingBoxProcessing.resize_image(
                                            upleft_points=_uplf_points, 
                                            obj_sizes=_obj_sizes, 
                                            before_img_size=self.cnet1.INPUT_SIZE, 
                                            after_img_size=_raw_img_sz,
                                            )
            return _uplf_points, _obj_sizes

        pred_heatmaps1, pred_obj_sizes1, pred_offsets1 = self.cnet1.predict(raw_images, need_conv_img=need_conv_img)
        pred_heatmaps2, pred_obj_sizes2, pred_offsets2 = self.cnet2.predict(raw_images, need_conv_img=need_conv_img)

        pred_heatmaps = (pred_heatmaps1 + pred_heatmaps2) * 0.5
        pred_obj_sizes = (pred_obj_sizes1 + pred_obj_sizes2) * 0.5
        pred_offsets = (pred_offsets1 + pred_offsets2) * 0.5

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = pred_bbox_one_sample(pred_heatmaps, 
                                                          pred_obj_sizes, 
                                                          pred_offsets, 
                                                          (raw_images.shape[1], raw_images.shape[0]),
                                                          )
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for hm, sz, ofs, raw_img in zip(pred_heatmaps, pred_obj_sizes, pred_offsets, raw_images):
                uplf_pt, obj_sz = pred_bbox_one_sample(hm, sz, ofs, (raw_img.shape[1], raw_img.shape[0]))
                uplf_points.append(uplf_pt)
                obj_sizes.append(obj_sz)

        return uplf_points, obj_sizes




class DetecPipeline_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3():
    def __init__(self, iou_thresh=None, score_thresh=None):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.cnet = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.croped_cnet = CropedCenternetPipeline_19100201Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet.load_model()
        self.croped_cnet.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        pred_upleft_points, pred_object_sizes = self.cnet.predict_bbox(raw_images, need_conv_img=need_conv_img)

        def _pred_bbox_one_sample(_img, _uplfs, _obj_szs):
            #visu.Visualization.visualize_pred_result(_img, _uplfs[0], _obj_szs[0])

            _croped_img = self.croped_cnet.crop_image_including_all_bbox(_img, _uplfs, _obj_szs, return_bbox=False)
            _pred_uplfs, _pred_obj_szs = self.croped_cnet.predict_bbox(_croped_img[np.newaxis], need_conv_img=need_conv_img)

            #visu.Visualization.visualize_pred_result(_croped_img, _pred_uplfs[0][0], _pred_obj_szs[0][0])

            _pred_uplfs, _pred_obj_szs = self.croped_cnet.inv_crop_image_including_all_bbox(_img.shape[1::-1], 
                                                                                            _uplfs, _obj_szs, 
                                                                                            _pred_uplfs[0], _pred_obj_szs[0])

            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            return _pred_uplfs, _pred_obj_szs

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = _pred_bbox_one_sample(raw_images, pred_upleft_points[0], pred_object_sizes[0])
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for imgs, p_uplf, p_obj_sz in zip(raw_images, pred_upleft_points, pred_object_sizes):
                uplfs, obj_szs = _pred_bbox_one_sample(imgs, p_uplf, p_obj_sz)
                uplf_points.append(uplfs)
                obj_sizes.append(obj_szs)

        #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

        return uplf_points, obj_sizes

class DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3():
    def __init__(self, iou_thresh=None, score_thresh=None, use_union_area=True):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.USE_UNION_AREA = use_union_area

        self.cnet = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.croped_cnet = CropedCenternetPipeline_19100201Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet.load_model()
        self.croped_cnet.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        pred_upleft_points, pred_object_sizes = self.cnet.predict_bbox(raw_images, need_conv_img=need_conv_img)

        def _pred_bbox_one_sample(_img, _uplfs, _obj_szs):
            #print('centernet')
            #visu.Visualization.visualize_pred_result(_img, _uplfs[0], _obj_szs[0])

            _croped_img = self.croped_cnet.crop_image_including_all_bbox(_img, _uplfs, _obj_szs, return_bbox=False)
            _pred_uplfs, _pred_obj_szs = self.croped_cnet.predict_bbox(_croped_img[np.newaxis], need_conv_img=need_conv_img)
            #visu.Visualization.visualize_pred_result(_croped_img, _pred_uplfs[0][0], _pred_obj_szs[0][0])

            _pred_uplfs, _pred_obj_szs = self.croped_cnet.inv_crop_image_including_all_bbox(_img.shape[1::-1], 
                                                                                            _uplfs, _obj_szs, 
                                                                                            _pred_uplfs[0], _pred_obj_szs[0])
            #print('croped centernet')
            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            # have bbox
            if _uplfs.shape[1] != 0:
                _pred_uplfs, _pred_obj_szs = self.nms_bbox([_uplfs[0], _pred_uplfs[0]], 
                                                           [_obj_szs[0], _pred_obj_szs[0]], 
                                                           self.IOU_THRESHOLD)
                _pred_uplfs, _pred_obj_szs = _pred_uplfs[np.newaxis], _pred_obj_szs[np.newaxis]

            #print('ens')
            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            return _pred_uplfs, _pred_obj_szs

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = _pred_bbox_one_sample(raw_images, pred_upleft_points[0], pred_object_sizes[0])
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for imgs, p_uplf, p_obj_sz in zip(raw_images, pred_upleft_points, pred_object_sizes):
                uplfs, obj_szs = _pred_bbox_one_sample(imgs, p_uplf, p_obj_sz)
                uplf_points.append(uplfs)
                obj_sizes.append(obj_szs)

        #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

        return uplf_points, obj_sizes

    def nms_bbox(self, upleft_points_list, object_sizes_list, iou_threshold):
        """
        Args:
            upleft_points_list: [upleft_points1, upleft_points2, ...], upleft_points shape = (num_box, 2)
            object_sizes_list: [object_sizes1, object_sizes2, ...], object_sizes shape = (num_box, 2)
        """
        uplfs = np.concatenate(upleft_points_list, axis=0)
        obj_szs = np.concatenate(object_sizes_list, axis=0)
        
        bottomrights = uplfs + obj_szs
        boxes = np.concatenate([uplfs, bottomrights], axis=1)

        scores = obj_szs[:,0] * obj_szs[:,1]

        nms_boxes = op_util.nms(boxes, scores, iou_threshold, use_union_area=self.USE_UNION_AREA)
        nms_uplfs = nms_boxes[:,:2]
        nms_obj_szs = nms_boxes[:,2:4] - nms_boxes[:,:2]

        return nms_uplfs, nms_obj_szs

class DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv():
    def __init__(self, iou_thresh=None, score_thresh=None, use_union_area=True):
        self.IOU_THRESHOLD = iou_thresh if iou_thresh is not None else 0.4
        self.SCORE_THRESHOLD = score_thresh if score_thresh is not None else 0.5
        
        self.USE_UNION_AREA = use_union_area

        self.cnet = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.croped_cnet = CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv2(iou_thresh=iou_thresh, score_thresh=score_thresh)
        return

    def load_model(self):
        self.cnet.load_model()
        self.croped_cnet.load_model()
        return

    def predict_bbox(self, raw_images, need_conv_img=True):
        pred_upleft_points, pred_object_sizes = self.cnet.predict_bbox(raw_images, need_conv_img=need_conv_img)

        def _pred_bbox_one_sample(_img, _uplfs, _obj_szs):
            #print('centernet')
            #visu.Visualization.visualize_pred_result(_img, _uplfs[0], _obj_szs[0])

            _croped_img = self.croped_cnet.crop_image_including_all_bbox(_img, _uplfs, _obj_szs, return_bbox=False)
            _pred_uplfs, _pred_obj_szs = self.croped_cnet.predict_bbox(_croped_img[np.newaxis], need_conv_img=need_conv_img)
            #visu.Visualization.visualize_pred_result(_croped_img, _pred_uplfs[0][0], _pred_obj_szs[0][0])

            _pred_uplfs, _pred_obj_szs = self.croped_cnet.inv_crop_image_including_all_bbox(_img.shape[1::-1], 
                                                                                            _uplfs, _obj_szs, 
                                                                                            _pred_uplfs[0], _pred_obj_szs[0])
            #print('croped centernet')
            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            # have bbox
            if _uplfs.shape[1] != 0:
                _pred_uplfs, _pred_obj_szs = self.nms_bbox([_uplfs[0], _pred_uplfs[0]], 
                                                           [_obj_szs[0], _pred_obj_szs[0]], 
                                                           self.IOU_THRESHOLD)
                _pred_uplfs, _pred_obj_szs = _pred_uplfs[np.newaxis], _pred_obj_szs[np.newaxis]

            #print('ens')
            #visu.Visualization.visualize_pred_result(_img, _pred_uplfs[0], _pred_obj_szs[0])

            return _pred_uplfs, _pred_obj_szs

        # one sample
        if len(raw_images.shape) == 3:
            uplf_points, obj_sizes = _pred_bbox_one_sample(raw_images, pred_upleft_points[0], pred_object_sizes[0])
        # some sample
        else:
            uplf_points, obj_sizes = [], []
            for imgs, p_uplf, p_obj_sz in zip(raw_images, pred_upleft_points, pred_object_sizes):
                uplfs, obj_szs = _pred_bbox_one_sample(imgs, p_uplf, p_obj_sz)
                uplf_points.append(uplfs)
                obj_sizes.append(obj_szs)

        #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

        return uplf_points, obj_sizes

    def nms_bbox(self, upleft_points_list, object_sizes_list, iou_threshold):
        """
        Args:
            upleft_points_list: [upleft_points1, upleft_points2, ...], upleft_points shape = (num_box, 2)
            object_sizes_list: [object_sizes1, object_sizes2, ...], object_sizes shape = (num_box, 2)
        """
        uplfs = np.concatenate(upleft_points_list, axis=0)
        obj_szs = np.concatenate(object_sizes_list, axis=0)
        
        bottomrights = uplfs + obj_szs
        boxes = np.concatenate([uplfs, bottomrights], axis=1)

        scores = obj_szs[:,0] * obj_szs[:,1]

        nms_boxes = op_util.nms(boxes, scores, iou_threshold, use_union_area=self.USE_UNION_AREA)
        nms_uplfs = nms_boxes[:,:2]
        nms_obj_szs = nms_boxes[:,2:4] - nms_boxes[:,:2]

        return nms_uplfs, nms_obj_szs


class ResNetPipeline_190913AspectVer3():
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190913_aspect_ver3', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                                  num_class=self.NUM_CLASS, 
                                                  resnet_version=self.RESNET_VERSION,
                                                  other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_190921AspectVer4():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing(128) -> gaussian filter -> median filter
    メモ：処理速度遅い(train, valデータの読み込みと処理に3時間ぐらいかかる)
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190921_aspect_ver4', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512
        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(5)
        #val_data_idxes = np.arange(5, 10)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_Ensemble190922_190913AspectVer3_190921AspectVer4():
    def __init__(self):
        self.myresnet1 = ResNetPipeline_190913AspectVer3()
        self.myresnet2 = ResNetPipeline_190921AspectVer4()
        return

    def load_model(self):
        self.myresnet1.load_model()
        self.myresnet2.load_model()
        return

    def predict(self, raw_images, soft=False):
        oups1 = self.myresnet1.predict(raw_images, soft=True)
        oups2 = self.myresnet2.predict(raw_images, soft=True)

        oups = (oups1 + oups2) * 0.5

        if not soft:
            # output number
            oups = np.argmax(oups, axis=1)

        return oups

class ResNetPipeline_190925AspectVer5():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing(255) -> gaussian filter -> median filter
    メモ：処理速度遅い(train, valデータの読み込みと処理に3時間ぐらいかかる)
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190925_aspect_ver5', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            _conv_img = _img

            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=255)
            
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            
            # binaryzation
            _conv_img = image_proc.ImageProcessing.binarize(_conv_img, method='mean')
            # opening
            _conv_img = image_proc.ImageProcessing.opening(_conv_img, kernelsize=3)
            _conv_img = image_proc.ImageProcessing.opening(_conv_img, kernelsize=3)
            #
            _conv_img = _conv_img[:,:,np.newaxis]

            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512
        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(5)
        #val_data_idxes = np.arange(5, 10)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_190926AspectVer6():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190926_aspect_ver6', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_190926AspectVer6_tta_test():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190926_aspect_ver6', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.05
        self.TTA_HIGHT_SHIFT_RANGE = 0.05
        self.TTA_ZOOM_RATE = None
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def run_eval(self):
        SEED = 2020

        # use indexs
        #tr_data_idxes = self.__use_indexes(use_train_data=True, 
        #                                   use_val_data=False, 
        #                                   seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)
        #val_data_idxes = val_data_idxes[:10]

        # data
        # image
        #tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        #tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        #tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        #self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        #
        self.load_model()

        # eval
        #pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input, need_print=True)
        
        #self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return


    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        oups = self.__predict_using_input(inputs, soft=soft)

        return oups

    def __predict_using_input(self, inputs, soft=False, need_print=False):
        # tta class instance
        #tta_cls_inst = classification_tta.TranslateAugmentation_9case(
        #                image_size_hw=self.INPUT_SIZE, 
        #                width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
        #                height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
        #                )
        tta_cls_inst = classification_tta.Augmentation_Translate8case_Zoom4case(
                        shift_rate_wh=[self.TTA_WIDTH_SHIFT_RANGE, self.TTA_HIGHT_SHIFT_RANGE], 
                        zoom_rate=self.TTA_ZOOM_RATE,
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_cls_inst.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft,
                                          need_print=need_print)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_190927AspectVer7():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing, mixup
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190927_aspect_ver7', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        MIXUP_ALPHA = 0.2

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   mixup_alpha=MIXUP_ALPHA,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #tr_data_idxes = np.arange(10)
        #val_data_idxes = np.arange(10, 20)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7():
    def __init__(self):
        self.myresnet1 = ResNetPipeline_190926AspectVer6()
        self.myresnet2 = ResNetPipeline_190927AspectVer7()
        return

    def load_model(self):
        self.myresnet1.load_model()
        self.myresnet2.load_model()
        return

    def predict(self, raw_images, soft=False):
        oups1 = self.myresnet1.predict(raw_images, soft=True)
        oups2 = self.myresnet2.predict(raw_images, soft=True)

        oups = (oups1 + oups2) * 0.5

        if not soft:
            # output number
            oups = np.argmax(oups, axis=1)

        return oups

class ResNetPipeline_Ensemble190929_190913AspectVer3_190921AspectVer4_190926AspectVer6_190927AspectVer7():
    def __init__(self):
        self.myresnet1 = ResNetPipeline_190913AspectVer3()
        self.myresnet2 = ResNetPipeline_190921AspectVer4()
        self.myresnet3 = ResNetPipeline_190926AspectVer6()
        self.myresnet4 = ResNetPipeline_190927AspectVer7()

        return

    def load_model(self):
        self.myresnet1.load_model()
        self.myresnet2.load_model()
        self.myresnet3.load_model()
        self.myresnet4.load_model()
        return

    def predict(self, raw_images, soft=False):
        oups1 = self.myresnet1.predict(raw_images, soft=True)
        oups2 = self.myresnet2.predict(raw_images, soft=True)
        oups3 = self.myresnet3.predict(raw_images, soft=True)
        oups4 = self.myresnet4.predict(raw_images, soft=True)

        oups = (oups1 + oups2 + oups3 + oups4) /4.0

        if not soft:
            # output number
            oups = np.argmax(oups, axis=1)

        return oups

class ResNetPipeline_191001AspectVer8_remove_few_label_test():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver2'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test191001_aspect_ver8', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        self.CUMNUM_LABEL_RATE_THRESH = 0.998 #0.99

        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __remove_few_num_label(self, one_hot_labels):

        label_num = np.sum(one_hot_labels, axis=0)

        sort_num = np.sort(label_num)[::-1]
        sort_label = np.argsort(label_num)[::-1]

        lot_label = sort_label[np.cumsum(sort_num)/np.sum(sort_num) < self.CUMNUM_LABEL_RATE_THRESH]

        lot_one_hot = keras.utils.to_categorical(lot_label, len(one_hot_labels[0]))
        lot_vect = np.sum(lot_one_hot, axis=0)

        lot_idx = np.arange(len(one_hot_labels))[np.sum(one_hot_labels * lot_vect, axis=1) != 0]

        return lot_idx

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 100 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        test_num = 300
        tr_data_idxes = np.random.choice(tr_data_idxes, test_num, replace=False)
        val_data_idxes = np.random.choice(val_data_idxes, test_num, replace=False)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # remove few num label
        if self.CUMNUM_LABEL_RATE_THRESH is not None:
            lot_idx = self.__remove_few_num_label(tr_letter_no_onehot)
            # image
            tr_inputs[0] = (tr_inputs[0])[lot_idx]
            tr_inputs[1] = (tr_inputs[1])[lot_idx]

            # letter no
            tr_letter_nos = tr_letter_nos[lot_idx]
            tr_letter_no_onehot = tr_letter_no_onehot[lot_idx]

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_190928AspectVer_focal_class_balance_test():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver1'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test190928_aspect_focal_class_balance_test', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}
        MIXUP_ALPHA = None
        
        CLASS_BALANCE_LOSS_EFFECTIVE_BETA = None #0.999
        FOCAL_LOSS_GAMMA = None #1.5

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   class_balance_loss_effective_beta=CLASS_BALANCE_LOSS_EFFECTIVE_BETA,
                                   focal_loss_gamma=FOCAL_LOSS_GAMMA,
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   mixup_alpha=MIXUP_ALPHA,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        test_num = 100
        tr_data_idxes = tr_data_idxes[:test_num]
        val_data_idxes = val_data_idxes[:test_num]

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_191009AspectVer9_struct_test():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (96, 96)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver2'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test191009_aspect_ver9_struct_test', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_and_conv_img2(self, indexes, need_print=False):
        imgs = []
        log_aspects = []

        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)

                #if imgs is None:
                #    imgs = conv_imgs_log_aspects[0]
                #else:
                #    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                #if log_aspects is None:
                #    log_aspects = conv_imgs_log_aspects[1]
                #else:
                #    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

                imgs.append(conv_imgs_log_aspects[0])
                log_aspects.append(conv_imgs_log_aspects[1])

        imgs = np.concatenate(imgs)
        log_aspects = np.concatenate(log_aspects)

        if need_print:
            print()

        return [imgs, log_aspects]


    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 100 # 200
        BATCH_SIZE = 256

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        test_num = 300
        tr_data_idxes = np.random.choice(tr_data_idxes, test_num, replace=False)
        val_data_idxes = np.random.choice(val_data_idxes, test_num, replace=False)

        # data
        # image
        tr_inputs = self.__read_and_conv_img2(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img2(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx


class ResNetPipeline_191002AspectVer9():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver2'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test191002_aspect_ver9', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2020

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #test_num = 10
        #tr_data_idxes = np.random.choice(tr_data_idxes, test_num, replace=False)
        #val_data_idxes = np.random.choice(val_data_idxes, test_num, replace=False)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_191002AspectVer9sv2():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (64, 64)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver2'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test191002_aspect_ver9sv2', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False):
        imgs = None
        log_aspects = None
        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs)
                if imgs is None:
                    imgs = conv_imgs_log_aspects[0]
                else:
                    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                if log_aspects is None:
                    log_aspects = conv_imgs_log_aspects[1]
                else:
                    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 512

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   save_file=None, 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def run_train(self):
        SEED = 2021

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #test_num = 10
        #tr_data_idxes = np.random.choice(tr_data_idxes, test_num, replace=False)
        #val_data_idxes = np.random.choice(val_data_idxes, test_num, replace=False)

        # data
        # image
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS)
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS)

        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.__predict_using_input(tr_inputs)
        pred_val_letter_nos = self.__predict_using_input(val_input)
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2():
    def __init__(self):
        self.myresnet1 = ResNetPipeline_191002AspectVer9()
        self.myresnet2 = ResNetPipeline_191002AspectVer9sv2()
        return

    def load_model(self):
        self.myresnet1.load_model()
        self.myresnet2.load_model()
        return

    def predict(self, raw_images, soft=False):
        oups1 = self.myresnet1.predict(raw_images, soft=True)
        oups2 = self.myresnet2.predict(raw_images, soft=True)

        oups = (oups1 + oups2) * 0.5

        if not soft:
            # output number
            oups = np.argmax(oups, axis=1)

        return oups

class ResNetPipeline_191009PseudoVer10():
    """
    image preprocessing: gausiann filter -> gamma correction -> ben's preprocessing -> gaussian filter -> median filter
    random erasing
    pseudo
    """
    def __init__(self):
        self.__config()
        self.__initilize()

        return

    def __config(self):
        self.INPUT_SIZE = (96, 96)
        self.INPUT_SHAPE = self.INPUT_SIZE[::-1] + (1,)
        self.OTHTER_INPUT_SHAPE = (1,)
        self.NUM_CLASS = len(kzsj_data.KuzushijiDataSet().get_letter_number_dict()[0])

        self.RESNET_VERSION = 'ver2'

        self.MODEL_DIR = os.path.join('.', 'result_recog', 'test191009_pseudo_ver10', 'my_resnet')
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, 'trained_model.h5')
        
        self.TTA_WIDTH_SHIFT_RANGE = 0.1
        self.TTA_HIGHT_SHIFT_RANGE = 0.1
        
        self.PSEUDO_SCORE_THRESH = 0.998
        return

    def __initilize(self):
        self.__build_model_instance()
        return

    def __conv_data_to_input(self, raw_images, do_normalize=True):
        # Conversion function
        def conv_func_one_sample(_img):
            #visu.Visualization.visualize_gray_img(_img)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_img, karnelsize=5)
            # gamma_correction
            GAMMA = 0.7
            _conv_img = image_proc.ImageProcessing.gamma_correction(_conv_img,
                                                                    gamma=GAMMA, 
                                                                    strength_criteria_is_0=True, 
                                                                    linear=True, 
                                                                    to_uint8=True)
            # ben's preprocessing
            _conv_img = image_proc.ImageProcessing.ben_preprocessing(_conv_img, base=128)
            # gaussian filter
            _conv_img = image_proc.ImageProcessing.gaussian_filter(_conv_img, karnelsize=5)
            # median filter
            _conv_img = image_proc.ImageProcessing.median_filter(_conv_img, karnelsize=5)
            # resize
            _conv_img = image_proc.ImageProcessing.resize(image=_conv_img, 
                                                          to_size=self.INPUT_SIZE, 
                                                          keep_aspect_ratio=False)
            #visu.Visualization.visualize_gray_img(_conv_img)
            # normalize
            if do_normalize:
                _conv_img = (_conv_img.astype('float32') - 127.5) / 127.5

            # aspect w/h
            _log_aspect_wh = np.log(_img.shape[1] / _img.shape[0])

            return _conv_img, _log_aspect_wh

        # shape = (H,W,C)
        if len(raw_images.shape) == 3:
            conv_imgs, log_aspects = conv_func_one_sample(raw_images)
        # shape = (num_sampel,H,W,C)
        else:
            conv_imgs = []
            log_aspects = []
            for img in raw_images:
                conv_img, log_aspect = conv_func_one_sample(img)
                conv_imgs.append(conv_img)
                log_aspects.append(log_aspect)
            conv_imgs = np.array(conv_imgs)
            log_aspects = np.array(log_aspects)

        return [conv_imgs, log_aspects]

    def __build_model_instance(self):
        self.my_resnet = resnet.MyResNet(image_shape=self.INPUT_SHAPE, 
                                         num_class=self.NUM_CLASS, 
                                         resnet_version=self.RESNET_VERSION,
                                         other_input_shape=self.OTHTER_INPUT_SHAPE)
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data, need_print=False):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=need_print)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=need_print)

        return imgs, ids

    def __read_train_upleftpoint_size(self, indexes=None):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        upleft_points, object_sizes = kzsj_dataset.read_train_upleftpoint_size(indexes)
        return upleft_points, object_sizes

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __read_and_conv_img(self, indexes, need_print=False, do_normalize=True):
        #imgs = None
        #log_aspects = None
        imgs = []
        log_aspects = []

        for idata in indexes:
            if need_print:
                if (idata+1) % 1 == 0:
                    print('\r read image {0}/{1}'.format(idata + 1, len(indexes)), end="")

            # read image
            img, _ = self.__read_img(indexes=idata, use_train_data=True, need_print=False)
            
            # read upleftpoint, size
            #upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            #object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            upleft_points, object_sizes = self.__read_train_upleftpoint_size(indexes=idata)

            # have bbox
            if len(upleft_points[0][0]) != 0:
                # crop image
                letter_imgs = self.__crop_img(img[0], upleft_points[0][0], object_sizes[0][0])

                # conv 
                conv_imgs_log_aspects = self.__conv_data_to_input(letter_imgs, do_normalize=do_normalize)
                #if imgs is None:
                #    imgs = conv_imgs_log_aspects[0]
                #else:
                #    imgs = np.concatenate([imgs, conv_imgs_log_aspects[0]], axis=0)
                #if log_aspects is None:
                #    log_aspects = conv_imgs_log_aspects[1]
                #else:
                #    log_aspects = np.concatenate([log_aspects, conv_imgs_log_aspects[1]], axis=0)

                imgs.append(conv_imgs_log_aspects[0])
                log_aspects.append(conv_imgs_log_aspects[1])

        imgs = np.concatenate(imgs)
        log_aspects = np.concatenate(log_aspects)

        if need_print:
            print()

        return [imgs, log_aspects]

    def __read_train_letter_no(self, idexes):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        letter_nos = kzsj_dataset.read_train_letter_no(idexes)

        flatten_letter_nos = []
        for ltno in letter_nos:
            flatten_letter_nos.extend(ltno.tolist())

        flatten_letter_nos = np.array(flatten_letter_nos)
        return flatten_letter_nos

    def __train(self, tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot):
        # buid model
        self.my_resnet.build_model()

        # save dir
        save_dir = self.MODEL_DIR
        if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        save_model_file = self.MODEL_FILE
        save_csv_file = os.path.join(save_dir, 'train_hist.csv')
        shutil.copy(__file__, save_dir)

        # training
        LEARNING_RATE = 0.001
        EPOCHS = 200 # 200
        BATCH_SIZE = 256

        RANDOM_ERASING_KWARGS = {'erasing_prob':0.5, 
                                 'area_rate_low':0.02, 
                                 'area_rate_high':0.4, 
                                 'aspect_rate_low':0.3, 
                                 'aspect_rate_high':3.3}

        NORMALIZE_BY_1275 = True

        # fit
        self.my_resnet.train_model(tr_inputs, tr_letter_no_onehot, 
                                   val_input, val_letter_no_onehot, 
                                   learning_rate=LEARNING_RATE, 
                                   epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   random_erasing_kwargs=RANDOM_ERASING_KWARGS,
                                   normalize_by_1275=NORMALIZE_BY_1275,
                                   save_file=save_model_file, #None 
                                   csv_file=save_csv_file)

        # save model
        self.my_resnet.save_model(save_file=save_model_file, only_model_plot=False)

        return

    def __eval(self, true_letter_no, pred_letter_no, save_file):
        acc = metrics.accuracy_score(true_letter_no, pred_letter_no)
        print(' acc : {0}'.format(acc))

        report = metrics.classification_report(true_letter_no, pred_letter_no)
        with open(save_file, mode='w') as f:
            f.write(report)

        return

    def __pseudo_labeling(self, score_thresh):
        # const
        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5
        USE_UNION_AREA = True
        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=False, 
                                        use_val_data=False, 
                                        seed=None)
        #data_idxes = np.arange(5)
        
        data_num = len(data_idxes)

        # loop of image data
        pseudo_labels = []
        pseudo_letter_imgs = []
        num_letters = 0

        print('pseudo labeling')
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, use_train_data=False)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pass
            else:
                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs[:6])

                # classify letter number
                pred_letter_no_one_hots = clsfy_pl.predict(letter_imgs, soft=True)
                num_letters = num_letters + len(pred_letter_no_one_hots)

                # pseudo
                pseudo_targets = np.max(pred_letter_no_one_hots, axis=1) > score_thresh
                pseudo_label = np.argmax(pred_letter_no_one_hots[pseudo_targets], axis=1)
                pseudo_letter_img = letter_imgs[pseudo_targets] # list

                pseudo_labels.extend(pseudo_label.tolist())
                pseudo_letter_imgs.extend(pseudo_letter_img)

                #if pseudo_labels is None:
                #    pseudo_labels = pseudo_label
                #    pseudo_letter_imgs = pseudo_letter_img
                #else:
                #    pseudo_labels = np.concatenate([pseudo_labels, pseudo_label], axis=0)
                #    pseudo_letter_imgs = np.concatenate([pseudo_letter_imgs, pseudo_letter_img], axis=0)
        print()
        pseudo_letter_imgs = np.array(pseudo_letter_imgs)
        pseudo_labels = np.array(pseudo_labels)

        num_pseudo_label = 0 if pseudo_labels is None else len(pseudo_labels)
        print('pseudo label num = {0} / {1}'.format(num_pseudo_label, num_letters))

        return pseudo_letter_imgs, pseudo_labels

    def run_train(self):
        SEED = 2022

        # pseudo label
        #pseudo_imgs, pseudo_letter_nos = self.__pseudo_labeling(score_thresh=self.PSEUDO_SCORE_THRESH)
        pseudo_imgs, pseudo_letter_nos = self.__pseudo_labeling(score_thresh=self.PSEUDO_SCORE_THRESH)
        pseudo_inputs = self.__conv_data_to_input(pseudo_imgs, do_normalize=False)
        del pseudo_imgs
        pseudo_letter_no_onehot = keras.utils.to_categorical(pseudo_letter_nos, self.NUM_CLASS)
        pseudo_letter_no_onehot = pseudo_letter_no_onehot.astype('float16')

        # use indexs
        tr_data_idxes = self.__use_indexes(use_train_data=True, 
                                           use_val_data=False, 
                                           seed=SEED)
        val_data_idxes = self.__use_indexes(use_train_data=True, 
                                            use_val_data=True, 
                                            seed=SEED)

        #test_num = 10
        #tr_data_idxes = np.random.choice(tr_data_idxes, test_num, replace=False)
        #val_data_idxes = np.random.choice(val_data_idxes, test_num, replace=False)

        # data
        # image (type(image)=uint8 to save memory)
        tr_inputs = self.__read_and_conv_img(tr_data_idxes, need_print=True, do_normalize=False)
        val_input = self.__read_and_conv_img(val_data_idxes, need_print=True, do_normalize=False)
        # letter no
        tr_letter_nos = self.__read_train_letter_no(tr_data_idxes)
        tr_letter_no_onehot = keras.utils.to_categorical(tr_letter_nos, self.NUM_CLASS).astype('float16')
        val_letter_nos = self.__read_train_letter_no(val_data_idxes)
        val_letter_no_onehot = keras.utils.to_categorical(val_letter_nos, self.NUM_CLASS).astype('float16')
        
        # concatenate tr pseudo
        tr_inputs[0] = np.concatenate([tr_inputs[0], pseudo_inputs[0]], axis=0)
        tr_inputs[1] = np.concatenate([tr_inputs[1], pseudo_inputs[1]], axis=0)
        tr_letter_nos = np.concatenate([tr_letter_nos, pseudo_letter_nos], axis=0)
        tr_letter_no_onehot = np.concatenate([tr_letter_no_onehot, pseudo_letter_no_onehot], axis=0)
        #
        del pseudo_inputs
        del pseudo_letter_nos
        del pseudo_letter_no_onehot
        
        # training
        self.__train(tr_inputs, tr_letter_no_onehot, val_input, val_letter_no_onehot)

        # eval
        pred_tr_letter_nos = self.predict(tr_inputs[0])
        pred_val_letter_nos = self.predict(val_input[0])
        
        self.__eval(tr_letter_nos, pred_tr_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_tr.txt'))
        self.__eval(val_letter_nos, pred_val_letter_nos, os.path.join(self.MODEL_DIR, 'result_report_val.txt'))

        return

    def load_model(self):
        self.my_resnet.load_model(self.MODEL_FILE)
        return

    def predict(self, raw_images, soft=False):
        inputs = self.__conv_data_to_input(raw_images)

        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __predict_using_input(self, inputs, soft=False):
        # tta class instance
        tta_trans9 = classification_tta.TranslateAugmentation_9case(
                        image_size_hw=self.INPUT_SIZE, 
                        width_shift_range=self.TTA_WIDTH_SHIFT_RANGE, 
                        height_shift_range=self.TTA_HIGHT_SHIFT_RANGE
                        )

        # predict
        oups = self.my_resnet.predict_tta(images=inputs[0], 
                                          tta_func=tta_trans9.augment_image, 
                                          other_inputs=inputs[1], 
                                          soft=soft)
        return oups

    def __stratify_train_test_split(self, labels, test_size_rate, random_state, shuffle):
        data_num = len(labels)

        # どのラベルが1個しか含まれていないか
        bin_counts = np.bincount(labels)
        unique_labels = np.nonzero(bin_counts)[0]
        unique_label_counts = bin_counts[unique_labels]
        one_labels = unique_labels[unique_label_counts==1]

        # 1個しか含まれていないラベルのインデックス、2個以上含まれているラベルのインデックス
        one_label_mask = np.in1d(labels, one_labels)
        one_label_idx = np.arange(len(labels))[one_label_mask]
        not_one_label_idx = np.arange(len(labels))[np.logical_not(one_label_mask)]

        # 2個以上含まれているラベルのインデックスをtrain, testに分割
        test_size = int(data_num * test_size_rate)
        train_idx, test_idx, _, _ = train_test_split(not_one_label_idx, 
                                                     labels[not_one_label_idx],
                                                     test_size=test_size, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle, 
                                                     stratify=labels[not_one_label_idx])
        # 1個しか含まれていないラベルのインデックスを結合
        train_idx = np.append(train_idx, one_label_idx)

        return train_idx, test_idx

class ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2_191009PseudoVer10():
    def __init__(self):
        self.myresnet1 = ResNetPipeline_191002AspectVer9()
        self.myresnet2 = ResNetPipeline_191002AspectVer9sv2()
        self.myresnet3 = ResNetPipeline_191009PseudoVer10()

        return

    def load_model(self):
        self.myresnet1.load_model()
        self.myresnet2.load_model()
        self.myresnet3.load_model()
        return

    def predict(self, raw_images, soft=False):
        oups1 = self.myresnet1.predict(raw_images, soft=True)
        oups2 = self.myresnet2.predict(raw_images, soft=True)
        oups3 = self.myresnet3.predict(raw_images, soft=True)

        oups = (oups1 + oups2 + oups3) * (1/3)

        if not soft:
            # output number
            oups = np.argmax(oups, axis=1)

        return oups



class RecognitionPipeline_091501():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self):
        USE_TRAIN_DATA = True
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        # detection
        detec_pl = CenternetPipeline_190909Ver2(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190913AspectVer3()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092101():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self):
        USE_TRAIN_DATA = False
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_19091801Ver2(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190913AspectVer3()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092201():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self):
        USE_TRAIN_DATA = False
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_19091801Ver2(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190921AspectVer4()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092202():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_19091801Ver2(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190922_190913AspectVer3_190921AspectVer4()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092301():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_19092201Ver2_tta(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190922_190913AspectVer3_190921AspectVer4()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092302():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190922_190913AspectVer3_190921AspectVer4()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092401():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_19092401Ver2sv2(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190922_190913AspectVer3_190921AspectVer4()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092601():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190925AspectVer5()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092701():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190926AspectVer6()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092801():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092901():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_092902():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190913AspectVer3_190921AspectVer4_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)


        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:

            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_093001():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19092901Ver1(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100101():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19092901Ver1(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100301():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100302():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble190929_190926AspectVer6_190927AspectVer7()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(30)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100801():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        USE_UNION_AREA = True

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(30)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100802():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        USE_UNION_AREA = False

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_19100201Ver3(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(30)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_100803():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        USE_UNION_AREA = True

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_101301():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        USE_UNION_AREA = True

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_191009PseudoVer10()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return

class RecognitionPipeline_101302():
    def __init__(self):
        return

    def __use_indexes(self, use_train_data, use_val_data, seed):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        # data num
        if use_train_data:
            data_num = kzsj_dataset.get_train_data_num()
        else:
            data_num = kzsj_dataset.get_test_data_num()

        # data idx
        if use_train_data:
            np.random.seed(seed)
            data_idxes = np.random.choice(data_num, int(data_num*0.8), replace=False)
            if use_val_data:
                data_idxes = np.setdiff1d(np.arange(data_num), data_idxes)
            data_idxes = np.sort(data_idxes)
        else:
            data_idxes = range(data_num)

        return data_idxes

    def __read_img(self, indexes, use_train_data):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()

        if use_train_data:
            imgs, ids = kzsj_dataset.read_train_image(indexs=indexes, to_gray=True, need_print=False)
        else:
            imgs, ids = kzsj_dataset.read_test_image(indexs=indexes, to_gray=True, need_print=False)

        return imgs, ids

    def __read_dict(self):
        kzsj_dataset = kzsj_data.KuzushijiDataSet()
        return kzsj_dataset.get_letter_number_dict()

    def __crop_img(self, img, upleft_points, obj_sizes):
        croped_imgs = image_proc.ImageProcessing.crop(img, 
                                                      upleft_points[:,0], 
                                                      upleft_points[:,1], 
                                                      obj_sizes[:,0], 
                                                      obj_sizes[:,1])
        return croped_imgs

    def __letter_number_to_unicode(self, letter_nos, inv_dict_cat):
        pred_unicodes = []
        for lt_no in letter_nos:
            pred_unicodes.append(inv_dict_cat[lt_no])
        return pred_unicodes

    def __make_pred_label_centerpoint(self, pred_unicodes, pred_center_points):
        pred_label_centerpoint = ''
        for pred_uni, pred_cp in zip(pred_unicodes, pred_center_points):
            pred_label_centerpoint += ' ' + pred_uni
            pred_label_centerpoint += ' ' + str(pred_cp[0])
            pred_label_centerpoint += ' ' + str(pred_cp[1])
        pred_label_centerpoint = pred_label_centerpoint[1:]

        return pred_label_centerpoint

    def __make_submission(self, image_ids, pred_label_centerpoints, comment='', calc_f1_score=True):
        # save dir
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.join('.', 'submission', now)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # submission file
        subm_df = pd.concat([pd.DataFrame(image_ids), pd.DataFrame(pred_label_centerpoints)], axis=1)
        subm_df.columns = ['image_id', 'labels']
        subm_df.to_csv(os.path.join(save_dir, now + '_' + comment + '_' + 'submission.csv'), header=True, index=False)

        # py file
        shutil.copy(__file__, save_dir)

        # f1 score
        if calc_f1_score:
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)
            
            # only bbox
            f1, results_in_each_page = kzsj_eval.KuzushijiF1.kuzushiji_f1_train(subm_df, with_lable=False)
            tps = [x['tp'] for x in results_in_each_page]
            fps = [x['fp'] for x in results_in_each_page]
            fns = [x['fn'] for x in results_in_each_page]
            result_df = pd.concat([subm_df, pd.DataFrame(tps), pd.DataFrame(fps), pd.DataFrame(fns)], axis=1)
            result_df.columns = ['image_id', 'labels', 'tp', 'fp', 'fn']
            result_df.to_csv(os.path.join(save_dir, 'result_bbox_f1_' + str(f1)[:7] + '.csv'), header=True, index=False)

        return

    def recognition(self, use_train_data=True):
        USE_TRAIN_DATA = use_train_data
        SEED = 2020
        DO_VALIDATION = True

        IOU_THRESHOLD = 0.4
        SCORE_THRESHOLD = 0.5

        USE_UNION_AREA = True

        EXPAND_BBOX_SIZE_WH = [0, 0]

        # detection
        detec_pl = DetecPipeline_Ens_CenternetPipeline_Ensemble190923_19091801Ver2_19091802Ver3_CropedCenternetPipeline_Ensemble191008_19100201Ver3_19100801Ver3sv(iou_thresh=IOU_THRESHOLD, score_thresh=SCORE_THRESHOLD, use_union_area=USE_UNION_AREA)
        detec_pl.load_model()

        # classification
        clsfy_pl = ResNetPipeline_Ensemble191008_191002AspectVer9_191002AspectVer9sv2_191009PseudoVer10()
        clsfy_pl.load_model()

        # use indexs
        data_idxes = self.__use_indexes(use_train_data=USE_TRAIN_DATA, 
                                        use_val_data=DO_VALIDATION, 
                                        seed=SEED)
        #data_idxes = np.arange(5)
        #data_idxes = data_idxes[51:55]
        data_num = len(data_idxes)

        # dict
        dict_cat, inv_dict_cat = self.__read_dict()

        # loop of image data
        image_ids = []
        pred_label_centerpoints = []
        for idata in data_idxes:
            # print
            if (idata+1) % 1 == 0:
                print('\r recog image {0}/{1}'.format(idata + 1, data_num), end="")

            # image
            img, img_id = self.__read_img(idata, USE_TRAIN_DATA)
            image_ids.append(img_id)

            # detection
            pred_upleft_points, pred_object_sizes = detec_pl.predict_bbox(img)
            pred_upleft_points, pred_object_sizes = pred_upleft_points[0][0], pred_object_sizes[0][0] # shape(1,1,num_bbox,2) -> (num_bbox,2)
            #visu.Visualization.visualize_pred_result(img[0], pred_upleft_points, pred_object_sizes)

            # bbox expand
            pred_upleft_points, pred_object_sizes = bbox_proc.BoundingBoxProcessing.expand_bbox(
                                                        pred_upleft_points, 
                                                        pred_object_sizes, 
                                                        expand_size_w=EXPAND_BBOX_SIZE_WH[0], 
                                                        expand_size_h=EXPAND_BBOX_SIZE_WH[1])

            # have no object
            if pred_upleft_points.shape[0] == 0:
                pred_label_centerpoint = ''
            else:
                # calc center point
                pred_center_points = np.maximum((pred_upleft_points + pred_object_sizes * 0.5).astype(int), 0)

                # crop image
                letter_imgs = self.__crop_img(img[0], pred_upleft_points, pred_object_sizes)
                #visu.Visualization.visualize_gray_img(letter_imgs)

                # classify letter number
                pred_letter_nos = clsfy_pl.predict(letter_imgs)

                # letter number convert to unicode
                pred_unicodes = [inv_dict_cat[x] for x in pred_letter_nos]

                # concatenate unicode and center point
                pred_label_centerpoint = self.__make_pred_label_centerpoint(pred_unicodes, pred_center_points)

            # result
            pred_label_centerpoints.append(pred_label_centerpoint)
        print()

        # make submission
        self.__make_submission(image_ids, pred_label_centerpoints, '', calc_f1_score=USE_TRAIN_DATA)

        return