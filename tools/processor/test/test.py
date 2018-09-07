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

    # def test_compute_new_dynamic_size(self):
    #     image = Image.open(self.img_path)
    #     # image = image.resize((1200, 500), Image.ANTIALIAS)
    #     # image.save("detection_1200_500.jpg", 'JPEG', quality=95)
    #     # image_np = load_image_into_numpy_array(image)
    #     new_size = pre._compute_new_dynamic_size(image, 600, 1024)
    #     print("new_size:", new_size)
    #
    # def test_ssd_pre(self):
    #     resized_inputs_np = pre.ssd_pre_np(self.img_path)
    #     resized_inputs_tf = pre.ssd_pre_tf(self.img_path)
    #     assert (resized_inputs_np.shape, resized_inputs_tf.shape)
    #     print("resized_inputs_np:", resized_inputs_np.shape)
    #
    # def test_faster_rcnn_pre(self):
    #     resized_inputs_np = pre.faster_rcnn_pre_np(self.img_path)
    #     resized_inputs_tf = pre.faster_rcnn_pre_tf(self.img_path)
    #     assert (resized_inputs_np.shape, resized_inputs_tf.shape)
    #     print("faster_rcnn resized_inputs_np:", resized_inputs_np.shape)

    def test_preprocessor(self):
        # result_ssd_np = pre.preprocessor(self.img_path, "ssd", "numpy")
        # result_ssd_tf = pre.preprocessor(self.img_path, "ssd", "tensorflow")
        # faster_rcnn_np = pre.preprocessor(self.img_path, "faster_rcnn", "numpy")
        faster_rcnn_tf = pre.preprocessor(self.img_path, "faster_rcnn", "tensorflow")

        # print("faster_rcnn_tf diff:", faster_rcnn_tf)
        # print("faster_rcnn diff:", faster_rcnn_np - faster_rcnn_tf)


if __name__ == "__main__":
    unittest.main()
