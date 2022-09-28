import argparse
from typing import Tuple
import cv2
import numpy as np
import onnxruntime

import nms


def run(weights, source, session=None, src_box_coords=True) -> np.ndarray:
    """
    Detect using ONNX model.
    :param weights: is the file path or serialized ONNX in a byte string
    :param source: is either the image path or an OpenCV image
    :param session: ONNX session
    :param src_box_coords: if true, the box coordinates will be relative to the source
    :return: a 2D array, each row is [left, top, right, bottom, conf]
    """

    if session == None:
        session = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider'])

    img0 = cv2.imread(source) if isinstance(source, str) else source
    img, (ratio, left_padding, top_padding) = resize_img(img0, 640)
    img = img.transpose((2, 0, 1))[::-1]  # hwc => chw, bgr => rgb
    img = np.ascontiguousarray(img).astype('float32')
    img /= 255

    img = img[None]  # Unsqueeze dim 0
    pred = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img})[0]

    pred_nms = nms.nms(np.squeeze(pred, axis=0), conf_thres=0.25, iou_thres=0.45)
    boxes = np.array(pred_nms)

    if src_box_coords:
        boxes_to_src_coords_inplace(boxes, img0.shape, (ratio, left_padding, top_padding))

    return boxes


def boxes_to_src_coords_inplace(boxes, src_shape, resize_config):
    if len(boxes) == 0:
        return

    ratio, left_padding, top_padding = resize_config
    boxes[:, [0, 2]] -= left_padding
    boxes[:, [1, 3]] -= top_padding
    boxes[:, :4] /= ratio

    # Clamp
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_shape[1])  # Left and right
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_shape[0])  # Top and bottom
    boxes[:, :4] = boxes[:, :4].round()


def resize_img(img, new_size) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resize an OpenCV image so its longest edge will be new_size.
    Then the image is padded so its width and height are multiples of 32
    Returns resized image, ratio, left padding and top padding
    """

    h0, w0, _ = img.shape
    h1 = new_size  # Height after resizing
    w1 = new_size  # Width after resizing
    if h0 >= w0:
        ratio = new_size / h0
        w1 = round(ratio * w0)
    else:
        ratio = new_size / w0
        h1 = round(ratio * h0)

    img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_LINEAR)

    stride = 32
    h2 = int((h1 + stride - 1) / stride) * stride
    w2 = int((w1 + stride - 1) / stride) * stride

    # Padding
    h_half_padding = (h2 - h1) // 2
    w_half_padding = (w2 - w1) // 2
    img = cv2.copyMakeBorder(img, top=h_half_padding,
                             bottom=h2 - h1 - h_half_padding,
                             left=w_half_padding,
                             right=w2 - w1 - w_half_padding,
                             borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))

    return img, (ratio, w_half_padding, h_half_padding)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model path')
    parser.add_argument('--source', type=str, help='image file')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
