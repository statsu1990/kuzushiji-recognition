"""
Hourglass network for keras
https://github.com/see--/keras-centernet/blob/master/keras_centernet/models/networks/hourglass.py
https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/master/src/net/hourglass.py
https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/master/src/net/hg_blocks.py
"""
import os

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
from keras.utils import plot_model

class StackedHourglassNet:
    def __init__(self, num_classes, image_shape, predict_width_hight, predict_offset, 
                 num_stacks, num_channels, num_channel_hg=[256, 384, 384, 384, 512]):
        self.NUM_CLASSES = num_classes
        self.IMAGE_SHAPE = image_shape # rgb=(y,x,3), gray=(y,x,1)

        self.PREDICT_WH = predict_width_hight # for centernet
        self.PREDICT_OFFSET = predict_offset # for centernet

        self.NUM_STACKS = num_stacks
        self.NUM_CHANNELS = num_channels
        self.NUM_CHANNEL_IN_HG = num_channel_hg

        return

    def create_model(self):
        """
        stacked hourglass network
            front module -> hourglass module -> ... -> hourglass module

        hourglass
            left half block -> bottom block -> right half block -> head module
                      ->->->-> connect  ->->->->
        """
        
        # input layer
        input = Input(shape=self.IMAGE_SHAPE)

        # fron module
        front_feat = self.__front_module(input)
        hg_feat = front_feat

        # loop of stack
        outputs = []
        for istack in range(self.NUM_STACKS):
            pre_hg_feat = hg_feat
            # hg feat and intermediate output
            hg_feat, inter_outputs = self.__hourglass_module(hg_feat)
            outputs.append(inter_outputs)
            
            hg_feat = self.__connect_pre_hg_to_now_hg(pre_hg_feat, hg_feat)
        if len(outputs) > 1:
            oups = Concatenate(axis=-1)(outputs)
        else:
            oups = outputs[0]

        #
        model = Model(inputs=input, outputs=oups)

        return model

    def __front_module(self, input):
        # front module, input to 1/4 resolution
        # 1 7x7 conv + maxpooling
        # 1 residual block
        _x = self.__convoluation_bn_relu(input, ksize=7, out_dim=128, stride=2)
        _x = self.__residual_block(_x, self.NUM_CHANNELS, stride=2)

        return _x
        
    def __hourglass_module(self, input):
        """
        left half block -> bottom block -> right half block -> head module
                      ->->->-> connect  ->->->->
        """
        # left half block: miniblock1, miniblock2, miniblock3, miniblock4 : 1, 1/2, 1/4 1/8 resolution
        left_feats = self.__left_half_blocks(input)

        # output of right half blocks
        right_feat = self.__right_half_blocks(left_feats)
        hg_feat = self.__convoluation_bn_relu(right_feat, ksize=3, out_dim=self.NUM_CHANNELS)

        # hourglass feature and output
        inter_outputs = self.__head_block(right_feat)

        return hg_feat, inter_outputs

    def __residual_block(self, input, num_out_channels, stride=1):
        # skip layer
        if K.int_shape(input)[-1] == num_out_channels and stride == 1:
            _skip = input
        else:
            _skip = self.__convoluation_bn(input, ksize=1, out_dim=num_out_channels, stride=stride)

        # residual
        _x = self.__convoluation_bn_relu(input, ksize=3, out_dim=num_out_channels, stride=stride)
        _x = self.__convoluation_bn(_x, ksize=3, out_dim=num_out_channels, stride=1)

        _x = Add()([_skip, _x])
        _x = Activation('relu')(_x)

        return _x

    def __left_half_blocks(self, input):
        # create left half blocks for hourglass module
        #   miniblock1 -> miniblock2 -> miniblock3 -> miniblock4 : 1, 1/2, 1/4 1/8 resolution

        left_feats = [input]

        for iblock, nc in enumerate(self.NUM_CHANNEL_IN_HG):
            _x = self.__residual_block(left_feats[-1], nc, stride=2)
            _x = self.__residual_block(_x, nc, stride=1)
            left_feats.append(_x)

        return left_feats

    def __right_half_blocks(self, left_feats):
        right_feat = self.__bottleneck_block(left_feats[-1], self.NUM_CHANNEL_IN_HG[-1])

        for iblock in reversed(range(len(self.NUM_CHANNEL_IN_HG))):
            right_feat = self.__connect_left_to_right(left_feats[iblock], right_feat, 
                                                      self.NUM_CHANNEL_IN_HG[iblock], 
                                                      self.NUM_CHANNEL_IN_HG[max(iblock - 1, 0)])

        return right_feat

    def __bottleneck_block(self, input, num_channel):
        _x = self.__residual_block(input, num_channel, stride=1)
        _x = self.__residual_block(_x, num_channel, stride=1)
        _x = self.__residual_block(_x, num_channel, stride=1)
        _x = self.__residual_block(_x, num_channel, stride=1)

        return _x

    def __connect_left_to_right(self, left, right, num_channels, num_channels_next):
        # left: 2 residual modules
        _x_left = self.__residual_block(left, num_channels_next)
        _x_left = self.__residual_block(_x_left, num_channels_next)

        # up: 2 times residual & nearest neighbour
        out = self.__residual_block(right, num_channels)
        out = self.__residual_block(out, num_channels_next)
        out = UpSampling2D()(out)
        out = Add()([_x_left, out])
        
        return out
        
    def __head_block(self, right_feat):
        """
        output two.
            No.1 : input of next hourglass network
            No.2 : intermediate output
        """
        _x = Conv2D(self.NUM_CHANNELS, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(right_feat)
        _x = BatchNormalization()(_x)

        # intermediate output
        inter_outputs = []
        
        def inter_oup(_xx, num_class, oup_activation):
            _oup = Conv2D(self.NUM_CHANNELS, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(_xx)
            _oup = Activation('relu')(_oup)
            _oup = Conv2D(num_class, kernel_size=(1, 1), padding='same')(_oup)

            _oup = Activation(oup_activation)(_oup)

            return _oup
        
        # heat
        heat_map = inter_oup(_x, self.NUM_CLASSES, oup_activation='sigmoid')
        inter_outputs.append(heat_map)

        # width, height
        if self.PREDICT_WH:
            width_height = inter_oup(_x, 2 * self.NUM_CLASSES, oup_activation='softplus')
            inter_outputs.append(width_height)

        # offset
        if self.PREDICT_OFFSET:
            offset = inter_oup(_x, 2 * self.NUM_CLASSES, oup_activation='sigmoid')
            inter_outputs.append(offset)

        inter_oups = Concatenate(axis=-1)(inter_outputs)

        return inter_oups

    def __connect_pre_hg_to_now_hg(self, pre_hg_ouptput, now_hg_output):
        _pre_x = Conv2D(self.NUM_CHANNELS, kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(pre_hg_ouptput)
        _pre_x = BatchNormalization()(_pre_x)

        _x = Conv2D(self.NUM_CHANNELS, kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(now_hg_output)
        _x = BatchNormalization()(_x)

        next_hg_input = Add()([_pre_x, _x])
        next_hg_input = Activation('relu')(next_hg_input)
        next_hg_input = self.__residual_block(next_hg_input, self.NUM_CHANNELS)

        return next_hg_input

    def __convoluation_bn(self, _x, ksize, out_dim, stride=1):
        _x = Conv2D(out_dim, ksize, strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal')(_x)
        _x = BatchNormalization()(_x)
        return _x

    def __convoluation_bn_relu(self, _x, ksize, out_dim, stride=1):
        _x = self.__convoluation_bn(_x, ksize, out_dim, stride)        
        _x = Activation('relu')(_x)
        return _x


def test190824():
    num_classes = 2
    image_shape = (512, 512, 3)
    predict_width_hight = True
    predict_offset = True
    
    num_stacks = 2
    num_channels = 256
    num_channel_hg=[num_channels, 64, 128]
    
    shg_net = StackedHourglassNet(num_classes, image_shape, predict_width_hight, predict_offset, num_stacks, num_channels, num_channel_hg)
    model = shg_net.create_model()
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    plot_model(model, to_file=os.path.join('.', 'stacked_hg_model.png'), show_shapes=True, show_layer_names=False)





