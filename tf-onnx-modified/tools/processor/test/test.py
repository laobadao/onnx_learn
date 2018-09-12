import tools.processor.preprocessor as pre
import unittest
import os
import numpy as np

np.set_printoptions(threshold=np.inf)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class PreprocessorTests(unittest.TestCase):

    def setUp(self):
        image_dir = "E:\\Intenginetech\\repos\\tensorflow-onnx\\tools\\processor\\"

        img_name = "detection_960_660.jpg"

        self.img_path = os.path.join(image_dir, img_name)
        self.model_config = "faster_rcnn"  # two types ssd or faster_rcnn


    def test_preprocessor(self):
        # result_ssd_np = pre.preprocessor(self.img_path, "ssd", "numpy")
        # result_ssd_tf = pre.preprocessor(self.img_path, "ssd", "tensorflow")
        # faster_rcnn_np = pre.preprocessor(self.img_path, "faster_rcnn", "numpy")
        faster_rcnn_tf = pre.preprocessor(self.img_path, "faster_rcnn", "tensorflow")

        # print("faster_rcnn_tf diff:", faster_rcnn_tf)
        # print("faster_rcnn diff:", faster_rcnn_np - faster_rcnn_tf)


if __name__ == "__main__":
    unittest.main()
