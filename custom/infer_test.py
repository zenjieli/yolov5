import unittest
from pathlib import Path
import os.path as osp
import cv2
import numpy as np

import custom.detect_with_onnx as detect_with_onnx
import nms


class infer_test(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

        FILE = Path(__file__).resolve()
        self.root = FILE.parents[1]

    def test_640x640(self):
        results = detect_with_onnx.run(
            osp.join(self.root, 'runs/train/yolov5s_hospital/weights/best.onnx'),
            osp.join(self.root, 'custom/data/VladB_Pedestrian_unsplash640x640.bmp'),
            src_box_coords=False
        )
        results_exp = [[225.61279, 61.05936, 391.26318, 598.37415, 0.93382],
                       [85.83737, 63.32065, 254.99301, 589.90881, 0.87988]]
        np.testing.assert_array_almost_equal(results_exp, results, 3)

    def test_632x465(self):
        results = detect_with_onnx.run(
            osp.join(self.root, 'runs/train/yolov5s_hospital/weights/best.onnx'),
            osp.join(self.root, 'custom/data/VladB_Pedestrian_unsplash.bmp'),
            src_box_coords=False
        )
        results_exp = [[339.384, 54.4870, 458.190, 438.619, 0.938205],
                       [234.830, 49.2765, 354.635, 438.630, 0.900235],
                       [76.4616, 124.806, 122.252, 218.475, 0.622002],
                       [0.421363, 117.518, 18.3557, 228.457, 0.278458]]

        np.testing.assert_array_almost_equal(results_exp, results, 3)

    def test_632x465_src_coords(self):
        results = detect_with_onnx.run(
            osp.join(self.root, 'runs/train/yolov5s_hospital/weights/best.onnx'),
            osp.join(self.root, 'custom/data/VladB_Pedestrian_unsplash.bmp'),
            src_box_coords=True
        )
        results_exp = [[335, 50, 452, 429, 0.93820],
                       [232, 45, 350, 429, 0.90024],
                       [76, 119, 121, 212, 0.62200],
                       [0, 112, 18, 222, 0.27846]]

        np.testing.assert_array_almost_equal(results_exp, results, 3)

    def test_resize_zero_image(self):
        img = np.zeros((290, 400, 3))
        img1, _ = detect_with_onnx.resize_img(img, 640)
        self.assertEqual((480, 640, 3), img1.shape)

    def test_resize(self):
        img = cv2.imread(osp.join(self.root, 'custom/data/VladB_Pedestrian_unsplash.bmp'))
        img1, _ = detect_with_onnx.resize_img(img, 640)

        img_exp = cv2.imread(osp.join(self.root, 'custom/data/VladB_Pedestrian_unsplash640x480.bmp'))
        np.testing.assert_array_equal(img_exp, img1)

    @staticmethod
    def read_list_from_file(filepath):
        f = open(filepath, 'r')
        lines = f.readlines()
        return [float(s) for s in lines]

    def test_nms(self):
        input_v = self.read_list_from_file('custom/data/VladB_Pedestrian_unsplash_yolo_output.txt')
        boxes = nms.nms(np.array(input_v).reshape(-1, 6))
        nms_result_exp = self.read_list_from_file('custom/data/VladB_Pedestrian_unsplash_nms_output.txt')

        boxes_flattened = []
        for box in boxes:
            boxes_flattened.append(box[0])
            boxes_flattened.append(box[1])
            boxes_flattened.append(box[2])
            boxes_flattened.append(box[3])
            boxes_flattened.append(box[4])

        np.testing.assert_array_almost_equal(np.array(nms_result_exp), np.array(boxes_flattened), 3)


if __name__ == '__main__':
    unittest.main()
