from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

IMG_PATH = "E:\\Intenginetech\\tf-onnx\\tools\\processor\\test\\detection_960_660.jpg"


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

img = Image.open(IMG_PATH)
img_np = load_image_into_numpy_array(img)
IMAGE_SIZE = (12, 8)
plt.figure(figsize=IMAGE_SIZE)
plt.imshow(img_np)
plt.show()