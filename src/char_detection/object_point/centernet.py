from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
import tensorflow as tf
import numpy as np

import os
import sys

from char_detection.object_point import hourglassnet as hgnet

# https://arxiv.org/abs/1904.07850

class CenterNet_SHGN:
    """
    Center Net using Stacked Hourglass Net
    """
    def __init__(self, num_classes, image_shape,  
                 num_stacks, num_channels, num_channel_hg, 
                 loss_coefs, hm_loss_alpha, hm_loss_beta):
        """
        loss_coefs = [loss lambda of heatmap, size, offset]
        """

        # config of target
        self.NUM_CLASSES = num_classes
        self.IMAGE_SHAPE = image_shape # rgb=(y,x,3), gray=(y,x,1)

        # config of stacked hourglass net
        self.NUM_STACKS = num_stacks
        self.NUM_CHANNELS = num_channels
        self.NUM_CHANNEL_IN_HG = num_channel_hg

        # config of loss
        self.LOSS_COEFS = loss_coefs
        self.HM_LOSS_ALPHA = hm_loss_alpha
        self.HM_LOSS_BETA = hm_loss_beta

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

    def build_model(self):
        shg_net = hgnet.StackedHourglassNet(num_classes=self.NUM_CLASSES, 
                                            image_shape=self.IMAGE_SHAPE, 
                                            predict_width_hight=True, 
                                            predict_offset=True, 
                                            num_stacks=self.NUM_STACKS, 
                                            num_channels=self.NUM_CHANNELS, 
                                            num_channel_hg=self.NUM_CHANNEL_IN_HG)
        self.model = shg_net.create_model()

        return

    def train_model(self, train_image, train_heatmap_objectsize_offset, 
                    val_image, val_heatmap_objectsize_offset, 
                    learning_rate, epochs, batch_size, lr_sche=None, 
                    save_file=None, csv_file=None):
        # compile
        opt = Adam(lr=learning_rate)
        self.model.compile(loss=self.__loss_stack_def, optimizer=opt)

        self.model.summary()

        # call back
        callbacks = []

        # learning rate scheduler
        if lr_sche is not None:
            lr_schedule = LearningRateScheduler(lr_sche)
        else:
            def lrs(_epoch):
                _lr = learning_rate
                if _epoch >= 50:
                    _lr = _lr * 0.1
                elif _epoch >= 100:
                    _lr = _lr * 0.01
                return _lr
            lr_schedule = LearningRateScheduler(lrs)
        callbacks.append(lr_schedule)

        # save model
        if save_file is not None:
            checkpoint = ModelCheckpoint(filepath=save_file,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(checkpoint)

        # save csv
        if csv_file is not None:
            csvlogger = CSVLogger(csv_file)
            callbacks.append(csvlogger)

        # fit
        hist = self.model.fit(x=train_image, y=train_heatmap_objectsize_offset, 
                              batch_size=batch_size, epochs=epochs, 
                              validation_data=(val_image, val_heatmap_objectsize_offset),
                              callbacks=callbacks,
                              shuffle=True
                              )
        return hist

    def train_model_with_generator(self, train_generator, 
                                   val_image, val_heatmap_objectsize_offset,
                                   train_steps_per_epoch, epochs, learning_rate, lr_sche=None, 
                                   save_file=None, csv_file=None):
        """
        """
        # compile
        opt = Adam(lr=learning_rate)
        self.model.compile(loss=self.__loss_stack_def, optimizer=opt)

        self.model.summary()

        # call back
        callbacks = []

        # learning rate scheduler
        if lr_sche is not None:
            lr_schedule = LearningRateScheduler(lr_sche)
        else:
            def lrs(_epoch):
                _lr = learning_rate
                if _epoch >= 50:
                    _lr = _lr * 0.1
                elif _epoch >= 100:
                    _lr = _lr * 0.01
                return _lr
            lr_schedule = LearningRateScheduler(lrs)
        callbacks.append(lr_schedule)

        # save model
        if save_file is not None:
            checkpoint = ModelCheckpoint(filepath=save_file,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(checkpoint)

        # save csv
        if csv_file is not None:
            csvlogger = CSVLogger(csv_file)
            callbacks.append(csvlogger)

        # train
        self.model.fit_generator(train_generator,
                                validation_data=(val_image, val_heatmap_objectsize_offset),
                                epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                                callbacks=callbacks, 
                                max_queue_size=5, 
                                workers=1, 
                                use_multiprocessing=False)

        return

    def train_model_with_generator_vlgene(self, train_generator, 
                                          val_generator,
                                          train_steps_per_epoch, epochs, learning_rate, lr_sche=None, 
                                          save_file=None, csv_file=None):
        """
        Uncomplimented
        """
        # compile
        opt = Adam(lr=learning_rate)
        self.model.compile(loss=self.__loss_stack_def, optimizer=opt)

        self.model.summary()


        # call back
        callbacks = []

        # learning rate scheduler
        if lr_sche is not None:
            lr_schedule = LearningRateScheduler(lr_sche)
        else:
            def lrs(_epoch):
                _lr = learning_rate
                if _epoch >= 50:
                    _lr = _lr * 0.1
                elif _epoch >= 100:
                    _lr = _lr * 0.01
                return _lr
            lr_schedule = LearningRateScheduler(lrs)
        callbacks.append(lr_schedule)

        # save model
        if save_file is not None:
            checkpoint = ModelCheckpoint(filepath=save_file,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(checkpoint)

        # save csv
        if csv_file is not None:
            csvlogger = CSVLogger(csv_file)
            callbacks.append(csvlogger)

        # train
        self.model.fit_generator(train_generator,
                                validation_data=val_generator,
                                epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                                callbacks=callbacks, 
                                max_queue_size=10, 
                                workers=1, 
                                use_multiprocessing=False)

        return

    def predict(self, images):
        """
        return heatmap, size(width, height), offset in heatmap
        """
        model_output = self.model.predict(images)[:, (self.NUM_STACKS-1)*3:]
        
        # heat map
        start = self.NUM_CLASSES * 5 * (self.NUM_STACKS - 1)
        end = start + self.NUM_CLASSES
        heatmap = model_output[:,:,:,start:end]
        # size
        start = end
        end = start + self.NUM_CLASSES * 2
        size = model_output[:,:,:,start:end]
        # offset
        start = end
        end = start + self.NUM_CLASSES * 2
        offset = model_output[:,:,:,start:end]
        
        return heatmap, size, offset

    def predict_tta(self, images, tta_cls_inst):
        """
        return heatmap, size(width, height), offset in heatmap
        """
        base_hms, base_szs, base_ofss = self.predict(images)

        # loop of data
        tta_hms = []
        tta_szs = []
        tta_ofss = []
        for img, base_hm, base_sz, base_ofs in zip(images, base_hms, base_szs, base_ofss):
            # augmented images
            aug_imgs = tta_cls_inst.augment_image(np.array([img]), 
                                                  np.array([base_hm]), 
                                                  np.array([base_sz]), 
                                                  np.array([base_ofs]))
            aug_imgs = aug_imgs[0]

            # augmented hm, sz, ofs
            if len(aug_imgs) != 0:
                aug_hms, aug_szs, aug_ofss = self.predict(aug_imgs)
            else:
                # have no bbox
                aug_hms, aug_szs, aug_ofss = None, None, None

            #
            tta_hm, tta_sz, tta_ofs = tta_cls_inst.integrate_heatmap_size_offset(img, 
                                                                                 base_hm, base_sz, base_ofs, 
                                                                                 aug_hms, aug_szs, aug_ofss)
            tta_hms.append(tta_hm)
            tta_szs.append(tta_sz)
            tta_ofss.append(tta_ofs)

        tta_hms = np.array(tta_hms)
        tta_szs = np.array(tta_szs)
        tta_ofss = np.array(tta_ofss)

        return tta_hms, tta_szs, tta_ofss

    def save_model(self, save_file, only_model_plot=False):
        """
        save model
        """
        # make dir
        save_dir = os.path.dirname(save_file)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # visualize
        plot_model(self.model, to_file=os.path.join(save_dir, 'model_structure.png'), show_shapes=True, show_layer_names=False)

        if not only_model_plot:
            # save model
            self.model.save(save_file)
            print('Saved trained model at %s ' % save_file)

        return

    def load_model(self, model_file):
        """
        load model .h5 file
        """
        self.model = load_model(model_file, custom_objects={'__loss_stack_def': self.__loss_stack_def})
        return

    def __loss_stack_def(self, y_true, y_pred):
        """
        y_true shape = (num_data, hm_x, hm_y, num_class + 2*num_class + 2*num_class)
        y_true[0,0,0,:] = [heatmap[0], ..., heatmap[C], 
                           width[0], hight[0] ..., width[C], hight[C], 
                           offset_x[0], offset_y[0], ..., offset_x[C], offset_y[C],
                           ]

        y_pred shape = (num_data, hm_x, hm_y, (num_class + 2*num_class + 2*num_class)*num_stack)
        y_pred[0,0,0,:] = [heatmap[0,0], ..., heatmap[0,C], 
                           width[0,0], hight[0,0] ..., width[0,C], hight[0,C], 
                           offset_x[0,0], offset_y[0,0], ..., offset_x[0,C], offset_y[0,C],
                           ...,
                           heatmap[S,0], ..., heatmap[S,C], 
                           width[S,0], hight[S,0] ..., width[S,C], hight[S,C], 
                           offset_x[S,0], offset_y[S,0], ..., offset_x[S,C], offset_y[S,C],
                           ]
          where C = num_class, S = num_stack
        """
        # loop of stacked hg
        ls_stack_det = 0
        num_in_one_stack = self.NUM_CLASSES * 5
        for istack in range(self.NUM_STACKS):
            start_channel = num_in_one_stack * istack
            end_channel = start_channel + num_in_one_stack

            ls_det = self.__loss_det(y_true, y_pred[:, :, :, start_channel:end_channel])
            ls_stack_det = ls_stack_det + ls_det
        
        return ls_stack_det

    def __loss_det(self, y_true, y_pred):
        """
        loss function
        
        y_true shape = (num_data, hm_x, hm_y, num_class + 2*num_class + 2*num_class)
        y_true[0,0,0,:] = [heatmap[0], ..., heatmap[C], 
                           width[0], hight[0] ..., width[C], hight[C], 
                           offset_x[0], offset_y[0], ..., offset_x[C], offset_y[C],
                           ]

        y_pred shape = (num_data, hm_x, hm_y, num_class + 2*num_class + 2*num_class)
        y_pred[0,0,0,:] = [heatmap[0], ..., heatmap[C], 
                           width[0], hight[0] ..., width[C], hight[C], 
                           offset_x[0], offset_y[0], ..., offset_x[C], offset_y[C],
                           ]
          where C = num_class
        """
        # is keypoint in heatmap
        start = 0
        end = start + self.NUM_CLASSES
        is_keypoint = self.__is_key_point(y_true[:,:,:,start:end]) # y_true[:,:,:,start:end] = heatmap
        # number of keypoint in heatmap
        num_keypoint = self.__num_key_point(is_keypoint)
        
        num_in_one_stack = self.NUM_CLASSES * 5

        # heat map loss
        start = 0
        end = start + self.NUM_CLASSES
        ls_heatmap = self.__loss_heatmap(y_true[:,:,:,start:end], y_pred[:,:,:,start:end], num_keypoint) # y_true and pred[:,:,:,start:end] = heatmap

        # size loss
        start = end
        end = start + self.NUM_CLASSES * 2
        ls_size = self.__loss_size(y_true[:,:,:,start:end], y_pred[:,:,:,start:end], num_keypoint, is_keypoint) # y_true and pred[:,:,:,start:end] = size

        # offset loss
        start = end
        end = start + self.NUM_CLASSES * 2
        ls_offset = self.__loss_offset(y_true[:,:,:,start:end], y_pred[:,:,:,start:end], num_keypoint, is_keypoint) # y_true and pred[:,:,:,start:end] = offset

        ls_det = self.LOSS_COEFS[0] * ls_heatmap + self.LOSS_COEFS[1] * ls_size + self.LOSS_COEFS[2] * ls_offset
        return ls_det

    def __loss_heatmap(self, hm_true, hm_pred, num_kp):
        # shape of heatmap = (hm_y, hm_x, num_class)
        # at keypoint
        epsilon = 1e-07 # K.epsilon
        ls_kp = (tf.math.sign(hm_true - 1.0 + epsilon) * 0.5 + 0.5) * K.pow(1.0 - hm_pred + epsilon, self.HM_LOSS_ALPHA) * K.log(hm_pred + epsilon)
        # not at keypoint
        ls_not_kp = K.pow(1.0 - hm_true + epsilon, self.HM_LOSS_BETA) * K.pow(hm_pred + epsilon, self.HM_LOSS_ALPHA) * K.log(1.0 - hm_pred + epsilon)
        # heatmap loss
        ls_hm = - K.sum(ls_kp + ls_not_kp, axis=(1,2,3)) / num_kp
        return ls_hm

    def __loss_size(self, size_true, size_pred, num_kp, is_kp):
        # shape of width_height = (hm_x, hm_y, (1+1)*num_class)
        #ls_size = K.sum(tf.matmul(is_kp, K.abs(size_true - size_pred)), axis=(1,2,3)) / num_kp
        ls_size = K.sum(is_kp * K.abs(size_true - size_pred), axis=(1,2,3)) / num_kp

        return ls_size

    def __loss_offset(self, offset_true, offset_pred, num_kp, is_kp):
        # shape of offset = (hm_x, hm_y, (1+1)*num_class)
        #ls_offset = K.sum(tf.matmul(is_kp, K.abs(offset_true - offset_pred)), axis=(1,2,3)) / num_kp
        ls_offset = K.sum(is_kp * K.abs(offset_true - offset_pred), axis=(1,2,3)) / num_kp
        return ls_offset

    def __is_key_point(self, true_heat_map):
        """
        if is keypoint = 1, else = 0
        """
        epsilon = 1e-07 # K.epsilon
        is_kp = tf.math.sign(true_heat_map - 1.0 + epsilon) * 0.5 + 0.5
        
        return is_kp
        
    def __num_key_point(self, is_keypoint):
        """
        number of keypoints in a image
        """
        num_keypoints = K.sum(is_keypoint, axis=(1,2,3))
        num_keypoints = K.clip(num_keypoints, 1.0, sys.float_info.max)

        return num_keypoints







