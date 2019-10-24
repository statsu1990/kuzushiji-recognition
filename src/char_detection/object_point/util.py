# https://qiita.com/ku2482/items/c0974359e575adbf3145

import numpy as np
 
def nms(boxes, scores, iou_threshold, score_threshold=float('-inf'), use_union_area=True):
    """
    Preprocess boxes with the non-maximal suppression. 
    
    boxes : [upper_left_x, upper_left_y, bottom_right_x, bottom_right_y] * num_box, shape=(num_box, 4)
    scores : [score] * num_box, shape=(num_box)
    iou_threshold : if overlap area > iou_threshold, suppress the bounding box.
    score_threshold : if score > score_threshold, suppress the bounding box.
    
    """
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []


    # sorted index by the score value
    idxs = np.argsort(scores)
    # index of score > score_threshold
    low_score_idxs = np.where(scores < score_threshold)[0]
    # remove index of score > score_threshold
    idxs = np.setdiff1d(idxs, low_score_idxs)

    while len(idxs) > 0:
        # get the index of the box with biggest score
        last = len(idxs) - 1
        # add the index of this box to the picked indexes
        pick.append(idxs[last])

        if len(idxs) != 1:
            box_last = boxes[idxs[last]]
            boxes_remain = boxes[idxs[:last]]

            # compute ious
            iou = _iou_one_box(box_last, boxes_remain, use_union_area=use_union_area)

            # delete
            large_iou_idxs = np.where(iou > iou_threshold)[0]
            idxs = np.delete(idxs, np.concatenate(([last], large_iou_idxs)))
        else:
            idxs = np.delete(idxs, last)

    return boxes[pick]

def iou_score(true_boxes, detected_boxes):
    """
    return average presicion iou, average recall iou, f value iou

    true_boxes : (upper left x, upper left y, bottom right x, bottom right y) * num_data
    detected_boxes : (upper left x, upper left y, bottom right x, bottom right y) * num_data
    """

    if len(true_boxes) == 0:
        ave_pres_iou = np.nan
        ave_recl_iou = np.nan
        f_iou = np.nan
    elif len(detected_boxes) == 0:
        ave_pres_iou = 0
        ave_recl_iou = 0
        f_iou = 0
    else:
        # target is detected boxes
        presicion_iou = [] # shape = (len(true_boxes))
        # target is true boxes
        recall_iou = np.zeros(len(true_boxes)) # shape = (len(detected_boxes))

        for detected_box in detected_boxes:
            iou = _iou_one_box(detected_box, true_boxes)

            presicion_iou.append(np.max(iou))
            recall_iou = np.maximum(recall_iou, iou)
        presicion_iou = np.array(presicion_iou)

        ave_pres_iou = np.average(presicion_iou)
        ave_recl_iou = np.average(recall_iou)
        # F値
        if abs(ave_pres_iou + ave_recl_iou) > 0.00001:
            f_iou = 2 * ave_pres_iou * ave_recl_iou / (ave_pres_iou + ave_recl_iou)
        else:
            f_iou = 0

    return ave_pres_iou, ave_recl_iou, f_iou

def _iou_one_box(base_box, comparison_boxes, use_union_area=True):
    """
    base_box : (upper left x, upper left y, bottom right x, bottom right y)
    comparison_boxes : (upper left x, upper left y, bottom right x, bottom right y) * num_data
    """
    # the maximum cordinates
    max_box = np.maximum(base_box, comparison_boxes)
    # the minimum cordinates
    min_box = np.minimum(base_box, comparison_boxes)

    # compute the common area
    common_area = np.multiply(
        np.maximum(0, min_box[:, 2] - max_box[:, 0] + 1),
        np.maximum(0, min_box[:, 3] - max_box[:, 1] + 1)
    )
    # compute the area of 'base_box'
    area_1 = np.multiply(
        np.maximum(0, base_box[2] - base_box[0] + 1),
        np.maximum(0, base_box[3] - base_box[1] + 1)
    )
    # compute the area of 'comparison_boxes'
    area_2 = np.multiply(
        np.maximum(0, comparison_boxes[:, 2] - comparison_boxes[:, 0] + 1),
        np.maximum(0, comparison_boxes[:, 3] - comparison_boxes[:, 1] + 1)
    )
    # compute the union area
    union_area = area_1 + area_2 - common_area
    # compute the IOUs
    if use_union_area:
        iou = common_area / union_area
    else:
        iou = np.maximum(common_area / area_1, common_area / area_2)

    return iou