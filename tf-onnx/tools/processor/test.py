import tools.processor.preprocessor as pre
import unittest
import os

class PreprocessorTests(unittest.TestCase):

    def setUp(self):
        img_name = "detection_3.jpg"
        image_dir = "E:\\Intenginetech\\repos\\tensorflow-onnx\\tools\\processor\\"
        self.img_path = os.path.join(image_dir, img_name)
        self.model_config = "ssd" # two types ssd or faster_rcnn

    def test_build_image_example(self):
        image_np_expanded = pre.build_image_example(self.img_path)
        print("image_np_expanded:", image_np_expanded.shape)






if __name__ == "__main__":
    unittest.main()
