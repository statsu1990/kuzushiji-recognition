import copy
import numpy as np
import cv2

class ImageProcessing:
    def __init__(self):
        return

    @staticmethod
    def crop(image, upleft_xs, upleft_ys, widths, heights):
        """
        crop image
        Args:
            image: one image(ndarray)
        Returns:
            croped images: num is len(upleft_xs)
        """
        croped_imgs = []

        for ul_x, ul_y, w, h in zip(upleft_xs, upleft_ys, widths, heights):
            start_y = np.maximum(int(ul_y), 0)
            end_y = np.minimum(int(ul_y) + int(np.ceil(h)), image.shape[0])

            start_x = np.maximum(int(ul_x), 0)
            end_x = np.minimum(int(ul_x) + int(np.ceil(w)), image.shape[1])

            croped_img = image[start_y:end_y, start_x:end_x]
            croped_imgs.append(croped_img)

        all_h_is_same = True
        for i, cr_img in enumerate(croped_imgs):
            if i == 0:
                pre_h = cr_img.shape[0]
            else:
                all_h_is_same = (pre_h == cr_img.shape[0])
            if not all_h_is_same:
                break

        if all_h_is_same:
            croped_imgs.append(np.array([]))
            croped_imgs = np.array(croped_imgs)
            croped_imgs = croped_imgs[:-1]
        else:
            croped_imgs = np.array(croped_imgs)
        return croped_imgs

    @staticmethod
    def resize(image, to_size, keep_aspect_ratio=False, interpolation='INTER_LINEAR'):
        """
        resize image
        Args:
            to_size: size of resized image (w, h)
            keep_aspect_ratio: keep aspect ratio or not
            interpolation: 'INTER_LINEAR', 'INTER_NEAREST'
            http://opencv.jp/opencv-2svn/cpp/geometric_image_transformations.html
        """
        if interpolation == 'INTER_NEAREST':
            intepl = cv2.INTER_NEAREST
        else:
            intepl = cv2.INTER_LINEAR

        size_w = to_size[0]
        size_h = to_size[1]

        if not keep_aspect_ratio:
            img_test = copy.copy(image)
            if len(image.shape) > 2:
                if image.shape[2] == 1:
                    img_test = image.reshape(image.shape[:2])

            resized_image = cv2.resize(img_test, dsize=to_size, interpolation=intepl)
            #resized_image = cv2.resize(image, dsize=to_size, interpolation=intepl)
            if len(resized_image.shape) == 2:
                resized_image = resized_image[:,:,np.newaxis]
        else:
            # calculate size after resize to keep aspect ratio
            image_w = image.shape[1]
            image_h = image.shape[0]
            # align w or not
            align_w = (size_w/image_w < size_h/image_h)
            if align_w:
                resize_w = size_w
                resize_h = int(size_w/image_w * image_h)
            else:
                resize_w = int(size_h/image_h * image_w)
                resize_h = size_h

            # resize
            resized_image = cv2.resize(image, dsize=(resize_w, resize_h), interpolation=intepl)
            if len(resized_image.shape) == 2:
                resized_image = resized_image[:,:,np.newaxis]

            # padding
            constant_values = np.average(resized_image)
            if align_w:
                p = (size_h - resize_h) // 2
                m = (size_h - resize_h) % 2
                resized_image = np.pad(resized_image, 
                                       pad_width=((p+m, p), (0, 0), (0, 0)),
                                       mode='constant', 
                                       constant_values=constant_values)
            else:
                p = (size_w - resize_w) // 2
                m = (size_w - resize_w) % 2
                resized_image = np.pad(resized_image, 
                                       pad_width=((0, 0), (p+m, p), (0, 0)),
                                       mode='constant', 
                                       constant_values=constant_values)
            
        return resized_image

    @staticmethod
    def zoom(images, zoom_rate, interpolation='INTER_LINEAR'):
        img_size_w = images.shape[2]
        img_size_h = images.shape[1]

        w_cut = int(img_size_w * zoom_rate / 2)
        h_cut = int(img_size_h * zoom_rate / 2)

        zoom_imgs = np.pad(images, 
                           pad_width=((0,0),
                                     (np.abs(h_cut), np.abs(h_cut)), 
                                     (np.abs(w_cut), np.abs(w_cut)), 
                                     (0, 0)), 
                           mode='edge')
        zoom_imgs = zoom_imgs[:, 
                              np.abs(h_cut)+h_cut : np.abs(h_cut)+img_size_h-h_cut,
                              np.abs(w_cut)+w_cut : np.abs(w_cut)+img_size_w-w_cut,
                              :]

        # resize
        #num_zoom_imgs = len(zoom_imgs)
        resized_imgs = []
        for i, zoom_img in enumerate(zoom_imgs):
            #print('\r zoom image {0}/{1}'.format(i + 1, num_zoom_imgs), end="")
            resized_imgs.append(ImageProcessing.resize(zoom_img, (img_size_w, img_size_h), keep_aspect_ratio=False, interpolation=interpolation))
            zoom_imgs = np.array(resized_imgs)
        print()

        return zoom_imgs

    @staticmethod
    def gamma_correction(images, gamma, strength_criteria_is_0=True, linear=False, to_uint8=True):
        """
        gamma correction.
        """
        max_strg = 255.0 #np.max(image)

        if not linear:
            if strength_criteria_is_0:
                cor_images = max_strg * (images / max_strg) ** gamma
            else:
                cor_images = max_strg * (1.0 - (1.0 - images / max_strg) ** (1.0 / gamma))
        else:
            if strength_criteria_is_0:
                cor_images = max_strg * np.minimum(1.0, (images / max_strg) * (1.0 / gamma))
            else:
                cor_images = max_strg * np.maximum(0.0, (images / max_strg - 1.0) * gamma + 1.0)

        if to_uint8:
            cor_images = np.uint8(cor_images)

        return cor_images

    @staticmethod
    def gaussian_filter(image, karnelsize):
        return cv2.GaussianBlur(image, (karnelsize, karnelsize), 0)

    @staticmethod
    def median_filter(image, karnelsize):
        return cv2.medianBlur(image, karnelsize)

    @staticmethod
    def translate(images, w_shift, h_shift, mode='edge'):
        """
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        Args:
            images: shape=(sample_num, H, W, C) or (H, W, C)
        """
        # shape=(sample_num, H, W, C)
        if len(images.shape) == 4:
            image_h = images.shape[1]
            image_w = images.shape[2]

            # padding
            pad_w_0 = (0, 0)
            pad_w_1 = (np.abs(h_shift), np.abs(h_shift))
            pad_w_2 = (np.abs(w_shift), np.abs(w_shift))
            pad_w_3 = (0, 0)
            translated_images = np.pad(images, pad_width=(pad_w_0, pad_w_1, pad_w_2, pad_w_3), mode=mode)

            #
            w_start = np.abs(w_shift) + w_shift
            w_end = np.abs(w_shift) + image_w + w_shift
            h_start = np.abs(h_shift) + h_shift
            h_end = np.abs(h_shift) + image_h + h_shift

            translated_images = translated_images[:, h_start:h_end, w_start:w_end, :]
        # shape=(H, W, C)
        else:
            image_h = images.shape[0]
            image_w = images.shape[1]

            # padding
            pad_w_1 = (np.abs(h_shift), np.abs(h_shift))
            pad_w_2 = (np.abs(w_shift), np.abs(w_shift))
            pad_w_3 = (0, 0)
            translated_images = np.pad(images, pad_width=(pad_w_1, pad_w_2, pad_w_3), mode=mode)

            #
            w_start = np.abs(w_shift) + w_shift
            w_end = np.abs(w_shift) + image_w + w_shift
            h_start = np.abs(h_shift) + h_shift
            h_end = np.abs(h_shift) + image_h + h_shift

            translated_images = translated_images[h_start:h_end, w_start:w_end, :]

        return translated_images

    @staticmethod
    def ben_preprocessing(image, base=128):
        return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, base)

    @staticmethod
    def adaptive_binarize(image, method='mean', karnelsize=5, c=0):
        """
        Args:
            method: 'mean', 'gaussian'
        """
        if method == 'mean':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, karnelsize, c)
        elif method == 'gaussian':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, karnelsize, c)

    @staticmethod
    def binarize(image, method='median'):
        """
        Args:
            method: 'median', 'mean', 'otsu'
        """
        if method == 'median':
            thresh = np.median(image)
            _, b_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
            return b_img
        elif method == 'mean':
            thresh = np.average(image)
            _, b_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
            return b_img
        elif method == 'otsu':
            _, b_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return b_img

    @staticmethod
    def opening(image, kernelsize):
        image = image.astype('uint8')
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernelsize,kernelsize),np.uint8))




