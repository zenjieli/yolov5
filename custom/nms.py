def iou(boxes, idx0, idx1):
    left0 = boxes[idx0][0]
    top0 = boxes[idx0][1]
    right0 = boxes[idx0][2]
    bottom0 = boxes[idx0][3]
    left1 = boxes[idx1][0]
    top1 = boxes[idx1][1]
    right1 = boxes[idx1][2]
    bottom1 = boxes[idx1][3]

    if left0 >= right0 or top0 >= bottom0 or left1 >= right1 or top1 >= bottom1:
        return 0  # Invalid input BBs, returns 0

    # The four corners of the intersection
    left2 = max(left0, left1)
    top2 = max(top0, top1)
    right2 = min(right0, right1)
    bottom2 = min(bottom0, bottom1)

    # No overlapping
    if left2 >= right2 or top2 >= bottom2:
        return 0

    intersection_area = (right2 - left2) * (bottom2 - top2)
    union_area = (right0 - left0) * (bottom0 - top0) + (right1 - left1) * (bottom1 - top1) - intersection_area

    return intersection_area / union_area


def get_bounding_boxes_above_thresh(boxes, conf_thres):
    """Every 6 items for a bounding box, x_center, y_center, width, height, object confidence, class confidence
    """
    v_above_thres = []

    for box in boxes:
        conf = box[4] * box[5]  # obj_conf * cls_conf
        if conf > conf_thres:
            x_center = box[0]
            y_center = box[1]
            width = box[2]
            height = box[3]
            bb = [
                x_center - width / 2,  # Left
                y_center - height / 2,  # Top
                x_center + width / 2,  # Right
                y_center + height / 2,  # Bottom
                conf]  # Conf

            v_above_thres.append(bb)

    return v_above_thres


def idx_max_conf(boxes):
    """Input is a vector of bounding boxes
    """

    assert len(boxes) > 0

    idx = 0
    for i, box in enumerate(boxes):
        if boxes[idx][4] < box[4]:
            idx = i

    return idx


def nms(input_v, conf_thres=0.25, iou_thres=0.45):
    """input_v is a 2D array where each row is a box
    """
    bb_candidates = get_bounding_boxes_above_thresh(input_v, conf_thres)

    bb_selected = []
    while len(bb_candidates) > 0:
        idx_max = idx_max_conf(bb_candidates)

        # One BB found; store it
        bb_selected.append(bb_candidates[idx_max])

        # Add the rest non-overlapping BBs to the candidate list
        bb_non_overlapping = []
        for i in range(len(bb_candidates)):
            if i != idx_max and iou(bb_candidates, idx_max, i) <= iou_thres:
                bb_non_overlapping.append(bb_candidates[i])

        # The non-overlappping BBs become candidates in the next iteration
        bb_candidates = bb_non_overlapping

    return bb_selected
