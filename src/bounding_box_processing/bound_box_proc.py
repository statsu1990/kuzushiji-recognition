import copy
import numpy as np

class BoundingBoxProcessing:
    def __init__(self):
        return

    @staticmethod
    def resize_image(upleft_points, obj_sizes, before_img_size, after_img_size):
        """
        Args:
            upleft_points, obj_sizes: shape is (num_class, num_box, 2) or (num_box, 2)
            before_img_size : [w, h]
            after_img_size : [w, h]
        """
        rescale = np.array(after_img_size) / np.array(before_img_size)

        def cov_resize(_objs, _rescale):
            if len(_objs.shape) == 2:
                # rescale
                _resized_objs = _objs * _rescale
            else:
                _resized_objs = []
                # loop of class
                for _obj_cls in _objs:
                    # have no object
                    if len(_obj_cls) == 0:
                        _resized_objs.append(copy.copy(_obj_cls))
                    else:
                        # rescale
                        _resized_obj_cls = _obj_cls * _rescale
                        _resized_objs.append(_resized_obj_cls)
                _resized_objs = np.array(_resized_objs)
            return _resized_objs

        # upper left point
        if upleft_points is not None:
            resized_uplfs = cov_resize(upleft_points, rescale)
        else:
            resized_uplfs = None

        # object size
        if upleft_points is not None:
            resized_obj_sizes = cov_resize(obj_sizes, rescale)
        else:
            resized_obj_sizes = None

        return resized_uplfs, resized_obj_sizes

    @staticmethod
    def transrate_image(upleft_points, shift_w, shift_h):
        """
        Args:
            upleft_points, : shape is (num_class, num_box, 2) or (num_box, 2)
        """
        shift_amount = np.array([shift_w, shift_h])

        if len(upleft_points.shape) == 2:
            # shift
            shifted_uplfs = upleft_points - shift_amount
        else:
            shifted_uplfs = []
            # loop of class
            for uplf_cls in upleft_points:
                # have no object
                if len(uplf_cls) == 0:
                    shifted_uplfs.append(copy.copy(uplf_cls))
                else:
                    # shift
                    shifted_uplf_cls = uplf_cls - shift_amount
                    shifted_uplfs.append(shifted_uplf_cls)
            shifted_uplfs = np.array(shifted_uplfs)
        
        return shifted_uplfs

    @staticmethod
    def crop_image(upleft_points, obj_sizes, crop_upleft_point, crop_size, remove_out_bbox=False):
        """
        Args:
            upleft_points, : shape is (num_class, num_box, 2) or (num_box, 2)
            obj_sizes, : shape is (num_class, num_box, 2) or (num_box, 2)
            remove_out_bbox: out bbox is that center point exists out of croped image.
        Returns:
            upleft_points: after croping
            obj_sizes: after croping
        """
        def _crop_img(_uplfs, _szs):
            """
            Args:
                upleft_points, : shape is (num_box, 2)
                obj_sizes, : shape is (num_box, 2)
                remove_out_bbox: out bbox is that center point exists out of croped image.
            """
            if len(_uplfs) != 0:
                _croped_uplfs = _uplfs - crop_upleft_point
                _croped_obj_szs = copy.copy(_szs)

                if remove_out_bbox:
                    _center = _croped_uplfs + 0.5 * _croped_obj_szs
                    _in_xrange = np.logical_and(_center[:,0] >= 0, _center[:,0] < crop_size[0])
                    _in_yrange = np.logical_and(_center[:,1] >= 0, _center[:,1] < crop_size[1])
                    _in_range = np.logical_and(_in_xrange, _in_yrange)

                    _croped_uplfs = _croped_uplfs[_in_range]
                    _croped_obj_szs = _croped_obj_szs[_in_range]
            else:
                _croped_uplfs = copy.copy(_uplfs)
                _croped_obj_szs = copy.copy(_szs)

            return _croped_uplfs, _croped_obj_szs

        #
        if len(upleft_points.shape) == 2:
            croped_uplfs, croped_obj_szs = _crop_img(upleft_points, object_sizes)
        else:
            croped_uplfs = []
            croped_obj_szs = []
            for uplf, sz in zip(upleft_points, obj_sizes):
                croped_uplf, croped_sz = _crop_img(uplf, sz)
                croped_uplfs.append(croped_uplf)
                croped_obj_szs.append(croped_sz)
            croped_uplfs = np.array(croped_uplfs)
            croped_obj_szs = np.array(croped_obj_szs)

        return croped_uplfs, croped_obj_szs

    @staticmethod
    def inv_crop_image(upleft_points, obj_sizes, croped_upleft_point):
        """
        Args:
            upleft_points, : shape is (num_class, num_box, 2) or (num_box, 2)
            obj_sizes, : shape is (num_class, num_box, 2) or (num_box, 2)
        Returns:
            upleft_points: before croping
            obj_sizes: before croping
        """

        def _inv_crop_img(_uplfs, _szs):
            """
            Args:
                upleft_points, : shape is (num_box, 2)
                obj_sizes, : shape is (num_box, 2)
                remove_out_bbox: out bbox is that center point exists out of croped image.
            """
            _inv_croped_uplfs = _uplfs + croped_upleft_point
            _inv_croped_obj_szs = copy.copy(_szs)
            return _inv_croped_uplfs, _inv_croped_obj_szs

        if len(upleft_points.shape) == 2:
            inv_croped_uplfs, inv_croped_obj_szs = _inv_crop_img(upleft_points, obj_sizes)
        else:
            inv_croped_uplfs = []
            inv_croped_obj_szs = []
            for uplf, sz in zip(upleft_points, obj_sizes):
                inv_croped_uplf, inv_croped_sz = _inv_crop_img(uplf, sz)
                inv_croped_uplfs.append(inv_croped_uplf)
                inv_croped_obj_szs.append(inv_croped_sz)
            inv_croped_uplfs = np.array(inv_croped_uplfs)
            inv_croped_obj_szs = np.array(inv_croped_obj_szs)

        return inv_croped_uplfs, inv_croped_obj_szs

    @staticmethod
    def expand_bbox(upleft_points, obj_sizes, expand_size_w, expand_size_h):
        """
        size expansion.
        Returns: 
            expanded upper left point
            expanded size : size = expand_size + size + expand_size
        """
        expanded_upleft_points = upleft_points - np.array([expand_size_w, expand_size_h])
        expanded_obj_sizes = obj_sizes + np.array([expand_size_w, expand_size_h]) * 2

        return expanded_upleft_points, expanded_obj_sizes

    @staticmethod
    def outermost_position(upleft_points, object_sizes):
        """
        Args:
            upleft_points, : shape is (num_class, num_box, 2) or (num_box, 2)
            object_sizes, : shape is (num_class, num_box, 2) or (num_box, 2)

        returns:
            [most left x, most up y, most right x, most bottom y]
             or
            [most left x, most up y, most right x, most bottom y] * num_class
        """
        def _calc_outmost_posi_wo_class(_uplfs, _szs):
            """
            Args:
                upleft_points, : shape is (num_box, 2)
                object_sizes, : shape is (num_box, 2)
            Returns:
                [most left x, most up y, most right x, most bottom y]
            """
            if len(_uplfs) != 0:
                # all object
                _most_left_x = np.min(_uplfs[:,0])
                _most_up_y = np.min(_uplfs[:,1])
                _most_right_x = np.max(_uplfs[:,0] + _szs[:,0])
                _most_bottom_y = np.max(_uplfs[:,1] + _szs[:,1])
                _outmost_posi = [_most_left_x, _most_up_y, _most_right_x, _most_bottom_y]
            else:
                _outmost_posi = []
            return _outmost_posi

        if len(upleft_points[0].shape) == 1:
            outermost_position = _calc_outmost_posi_wo_class(upleft_points, object_sizes)
        else:
            outermost_positions = []
            for uplf, obj_sz in zip(upleft_points, object_sizes):
                outermost_positions.append(_calc_outmost_posi_wo_class(uplf, obj_sz))

            outmostposis_not_empty = []
            for outmost in outermost_positions:
                if outmost != []:
                    outmostposis_not_empty.append(outmost)
            outmostposis_not_empty = np.array(outmostposis_not_empty)

            if len(outmostposis_not_empty) != 0:
                most_left_x = np.min(outmostposis_not_empty[:,0])
                most_up_y = np.min(outmostposis_not_empty[:,1])
                most_right_x = np.max(outmostposis_not_empty[:,2])
                most_bottom_y = np.max(outmostposis_not_empty[:,3])
                outermost_position = [most_left_x, most_up_y, most_right_x, most_bottom_y]
            else:
                outermost_position = []

        return outermost_position