import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

class ImageDataGeneratorWithOtherInputs(Sequence):
    """
    [Unimplemented]
    Data generator for center net.
    use this class if data size is large relative to the memory.

    reference:
        https://www.kumilog.net/entry/numpy-data-augmentation
        https://www.kumilog.net/entry/keras-generator
    """

    def __init__(self, 
                 images, other_inputs, y, 
                 batch_size, 
                 image_generator_args,
                 random_erasing_kwargs=None,
                 mixup_alpha=None,
                 normalize_by_1275=False,
                 ):
        self.IMAGES = images
        self.OTHER_INPUTS = other_inputs
        self.Y = y

        self.BATCH_SIZE = batch_size

        self.IMAGE_GENERATOR_ARGS = image_generator_args

        # {'erasing_prob':, 'area_rate_low':, 'area_rate_high':, 'aspect_rate_low':, 'aspect_rate_high':}
        self.RANDOM_ERASING_KWARGS = random_erasing_kwargs

        self.MIXUP_ALPHA = mixup_alpha

        self.NORMALIZE_BY_1275 = normalize_by_1275

        self.__initialize()

        return

    def __getitem__(self, idx):
        return self.__get_image_other_input(idx)

    def __len__(self):
        return self.STEP_NUM

    def on_epoch_end(self):
        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)

        return

    def __get_image_other_input(self, idx):
        batch_idxes = self.idxes[idx * self.BATCH_SIZE : (idx + 1) * self.BATCH_SIZE]

        # y
        ys = self.Y[batch_idxes]

        # other input
        if self.OTHER_INPUTS is not None:
            other_inputs = self.OTHER_INPUTS[batch_idxes]
        
        # get images
        if self.IMAGE_GENERATOR_ARGS is not None:
            images = next(self.image_datagen.flow(self.IMAGES[batch_idxes], batch_size=self.BATCH_SIZE, shuffle=False))
        else:
            images = self.IMAGES[batch_idxes]

        if self.NORMALIZE_BY_1275:
            images = (images.astype('float32') - 127.5) / 127.5

        # mixup
        if self.MIXUP_ALPHA is not None:
            mixup_rate = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA, len(batch_idxes))
            images = self.__mixup(mixup_rate, images)
            ys = self.__mixup(mixup_rate, ys)
            if self.OTHER_INPUTS is not None:
                other_inputs = self.__mixup(mixup_rate, other_inputs)

        # random erasing
        if self.RANDOM_ERASING_KWARGS is not None:
            images = self.__random_erasing(images, **self.RANDOM_ERASING_KWARGS)

        # other input
        if self.OTHER_INPUTS is None:
            return images, ys
        else:
            return [images, other_inputs], ys

    def __initialize(self):
        self.SAMPLE_NUM = len(self.IMAGES)
        self.STEP_NUM = int(np.ceil(self.SAMPLE_NUM / self.BATCH_SIZE))

        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)

        if self.IMAGE_GENERATOR_ARGS is not None:
            self.image_datagen = ImageDataGenerator(**self.IMAGE_GENERATOR_ARGS)
        return

    def __get_indexes(self, sample_num, do_shuffle=True):
        '''
        return shuffled indexes.
        '''
        indexes = np.arange(sample_num)
        if do_shuffle:
            indexes = np.random.permutation(indexes)
        return indexes

    def __random_erasing(self, images, erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        # https://qiita.com/takurooo/items/a3cba475a3db2c7272fe

        def _rand_erase(_img):
            target_image = _img.copy()

            if np.random.rand() > erasing_prob:
                # RandomErasingを実行しない
                return target_image 

            H, W, C = target_image.shape
            S = H * W

            while True:
                Se = np.random.uniform(area_rate_low, area_rate_high) * S # 画像に重畳する矩形の面積
                re = np.random.uniform(aspect_rate_low, aspect_rate_high) # 画像に重畳する矩形のアスペクト比

                He = int(np.sqrt(Se * re)) # 画像に重畳する矩形のHeight
                We = int(np.sqrt(Se / re)) # 画像に重畳する矩形のWidth

                xe = np.random.randint(0, W) # 画像に重畳する矩形のx座標
                ye = np.random.randint(0, H) # 画像に重畳する矩形のy座標

                if xe + We <= W and ye + He <= H:
                    # 画像に重畳する矩形が画像からはみ出していなければbreak
                    break

            #mask = np.random.randint(np.min(target_image), np.max(target_image), (He, We, C)) # 矩形がを生成 矩形内の値はランダム値
            mask = np.random.rand(He, We, C) # 矩形がを生成 矩形内の値はランダム値
            mask = mask * np.max(target_image) + (1 - mask) * np.min(target_image)
            
            target_image[ye:ye + He, xe:xe + We, :] = mask # 画像に矩形を重畳

            return target_image

        erased_imgs = []
        for img in images:
            erased_imgs.append(_rand_erase(img))
        erased_imgs = np.array(erased_imgs)

        return erased_imgs

    def __mixup(self, rate, x):
        rate_shape = list(x.shape)        
        for i in range(len(rate_shape)):
            if i > 0:
                rate_shape[i] = 1
        rate_shape = tuple(rate_shape)
        re_rate = rate.reshape(rate_shape)

        mixup_idx = self.__get_indexes(len(x), do_shuffle=True)

        #mix_x = re_rate * x + (1.0 - re_rate) * x[::-1]
        mix_x = re_rate * x + (1.0 - re_rate) * x[mixup_idx]
        return mix_x

