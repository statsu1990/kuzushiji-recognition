from PIL import Image, ImageDraw, ImageFont
from os import listdir
import os
import pandas as pd
import numpy as np

from kuzushiji_data import data_config

class KuzushijiDataSet:
    """
    データセットの取り扱い
    """
    def __init__(self):
        self.__config()
        return

    def __config(self):
        self.DATA_DIR = data_config.data_dir
        self.TRAIN_CSV = os.path.join(self.DATA_DIR, 'train.csv')
        self.UNICODE_TRANS_CSV = os.path.join(self.DATA_DIR, 'unicode_translation.csv')
        self.TRAIN_IMAGE_DIR = os.path.join(self.DATA_DIR, 'train_images')
        self.TEST_IMAGE_DIR = os.path.join(self.DATA_DIR, 'test_images')
        
        self.FONT_PATH = data_config.font_path
        return

    def read_train_image(self, indexs=None, to_gray=True, need_print=True):
        """
        学習用画像読み込み
        Returns:
            images
            image_ids
        """
        file_dir = self.TRAIN_IMAGE_DIR
        return KuzushijiDataSet.read_image(file_dir, indexs, to_gray, need_print)

    def read_test_image(self, indexs=None, to_gray=True, need_print=True):
        """
        テスト画像読み込み
        Returns:
            images
            image_ids
        """
        file_dir = self.TEST_IMAGE_DIR
        return KuzushijiDataSet.read_image(file_dir, indexs, to_gray, need_print)
    
    @staticmethod
    def read_image(file_dir, indexs=None, to_gray=True, need_print=True):
        """
        画像読み込み
        Returns:
            images
            image_ids
        """
        image_dir = file_dir
        image_files = np.array(os.listdir(image_dir))

        # use index
        use_idxs = indexs if indexs is not None else np.arange(len(image_files))
        image_files = image_files[use_idxs]
        if type(use_idxs) is int:
            image_files = [image_files]
        elif len(use_idxs.shape) == 0:
            image_files = [image_files]

        data_num = len(image_files)

        images = []
        image_ids = []
        # loop for image data
        for i, im_file in enumerate(image_files):
            if need_print:
                print('\r read image file {0}/{1}'.format(i + 1, data_num), end="")

            # read image
            if to_gray:
                image = Image.open(os.path.join(image_dir, im_file)).convert('L')
                image = np.asarray(image)[:,:,np.newaxis] # shape(H,W) -> shape(H,W,1)
            else:
                image = Image.open(os.path.join(image_dir, im_file)).convert('RGB')
                image = np.asarray(image) # shape(H,W,3)
            images.append(image)

            # image_id
            image_ids.append(os.path.splitext(im_file)[0])
        images = np.array(images)

        if need_print:
            print()

        return images, image_ids
    
    def get_train_data_num(self):
        image_dir = self.TRAIN_IMAGE_DIR
        image_files = np.array(os.listdir(image_dir))
        return len(image_files)

    def get_test_data_num(self):
        image_dir = self.TEST_IMAGE_DIR
        image_files = np.array(os.listdir(image_dir))
        return len(image_files)

    def read_train_upleftpoint_size(self, indexes=None):
        """
        args:

        returns:
            upleft_points : ndarray( [[x0,y0], [x1,y1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)
            object_sizes  : ndarray( [[w0,h0], [w1,h1], ...] * num_classes * num_data ), shape=(num_data, num_classes, num_keypoint, 2)

            where num_classes = 1
        """
        df_train = pd.read_csv(self.TRAIN_CSV)
        use_idxes = indexes if indexes is not None else np.arange(len(df_train))
        
        if df_train.iloc[use_idxes].size == 2:
            df_train = df_train.iloc[use_idxes:use_idxes+1]
        else:
            df_train = df_train.iloc[use_idxes]

        dt_num = len(df_train)
        
        #
        upleft_points = []
        object_sizes = []
        # loop of data
        for i in range(dt_num):
            img, labels = tuple(df_train.values[i])
            
            is_including_keypoint = type(labels) is str
            if is_including_keypoint:
                labels = np.array(labels.split(' ')).reshape(-1, 5)

                uplf_pt = []
                obj_sz = []
                for codepoint, x, y, w, h in labels:
                    uplf_pt.append([int(x), int(y)])
                    obj_sz.append([int(w), int(h)])
            
                # to ndarray
                uplf_pt = np.array(uplf_pt)
                obj_sz = np.array(obj_sz)

                # shape (num_keypoint, 2) -> (1(num_class), num_keypoint, 2)
                uplf_pt = uplf_pt[np.newaxis,:,:]
                obj_sz = obj_sz[np.newaxis,:,:]

                upleft_points.append(uplf_pt)
                object_sizes.append(obj_sz)
            else:
                upleft_points.append(np.empty((1,0,2)))
                object_sizes.append(np.empty((1,0,2)))

        return upleft_points, object_sizes
    
    def read_train_letter_no(self, indexes=None):
        df_train = pd.read_csv(self.TRAIN_CSV)
        use_idxes = indexes if indexes is not None else np.arange(len(df_train))
        
        if df_train.iloc[use_idxes].size == 2:
            df_train = df_train.iloc[use_idxes:use_idxes+1]
        else:
            df_train = df_train.iloc[use_idxes]

        dt_num = len(df_train)

        #
        dict_cat, inv_dict_cat = self.get_letter_number_dict()

        # loop of data
        letter_numbers = []
        for i in range(dt_num):
            _, labels = df_train.values[i]
            
            is_including_keypoint = type(labels) is str
            if is_including_keypoint:
                labels = np.array(labels.split(' ')).reshape(-1, 5)

                let_nos = []
                for codepoint, x, y, w, h in labels:
                    let_nos.append(dict_cat[codepoint])

                letter_numbers.append(np.array(let_nos))
            else:
                letter_numbers.append(np.array([]))

        letter_numbers = np.array(letter_numbers)
        return letter_numbers

    def get_letter_number_dict(self):
        """
        returns:
            letter number dict : {unicode : number}
            inv letter number dict : {number : unicode}
        """
        path_1 = self.TRAIN_CSV
        df_train=pd.read_csv(path_1)
        df_train=df_train.dropna(axis=0, how='any')#you can use nan data(page with no letter)
        df_train=df_train.reset_index(drop=True)

        category_names=set()

        for i in range(len(df_train)):
            ann = np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,x,y,width,height for each picture
            category_names=category_names.union({i for i in ann[:,0]})

        category_names = sorted(category_names)
        dict_cat = {list(category_names)[j] : j for j in range(len(category_names))}
        inv_dict_cat = {j : list(category_names)[j] for j in range(len(category_names))}

        return dict_cat, inv_dict_cat