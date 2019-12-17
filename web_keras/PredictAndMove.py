import keras
import cv2
from keras.models import load_model
import numpy as np
import sys

average = 82.1400742479


def LoadImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    img -= average
    img /= 255.
    return np.array(img[:, :, :3])


def abs_path(path):
    model = load_model(r"./vggmodel_65-0.00-1.00.hdf5")
    label_dict = {0: "air", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "drop", 7: "horse", 8: "ship",
                  9: "truck"}
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta()
                  , metrics=["accuracy"])
    class_pic = []
    # dir_list = [r'.\static\upload\taidi_2.jpg', r'.\static\upload\taidi_2.jpg']
    # for i in range(len(dir_list)):
    img = LoadImage(path)
    res = np.argmax(model.predict(np.array([img])))
    pic_class = label_dict[int(res)]
    print(pic_class)


if __name__ == "__main__":
    path = sys.argv[1]
    abs_path(path)
