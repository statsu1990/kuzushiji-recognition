from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self):
        return

    @staticmethod
    def visualize_pred_result(gray_image_ndarray, upleft_point, object_size):

        imsource = Image.fromarray(np.uint8(gray_image_ndarray[:,:,0])).convert('L')

        bbox_canvas = Image.new('L', imsource.size)
        bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
        
        for uplf, obj_sz in zip(upleft_point, object_size):
            x1 = int(uplf[0])
            y1 = int(uplf[1])
            x2 = int(x1 + obj_sz[0])
            y2 = int(y1 + obj_sz[1])

            bbox_draw.rectangle((x1, y1, x2, y2), fill=255, outline=0)
        
        imsource = Image.composite(imsource, bbox_canvas, Image.new("L", imsource.size, 128))
        
        img = np.asarray(imsource)
        plt.figure()
        plt.imshow(img)
        plt.gray()
        plt.show()
        plt.clf()

        return

    @staticmethod
    def visualize_gray_img(gray_images):
        def vis_one_sample(_img):
            plt.figure()
            plt.imshow(_img[:,:,0])
            plt.gray()
            plt.show()
            plt.clf()

        if len(gray_images.shape) == 3:
            vis_one_sample(gray_images)
        else:
            for img in gray_images:
                vis_one_sample(img)
        return