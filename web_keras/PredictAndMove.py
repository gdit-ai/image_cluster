import keras
import cv2
from keras.models import load_model
import numpy as np
import sys


def LoadImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    return np.array(img[:, :, :3])


def predict(path):
    model = load_model(r"vggmodel_21-0.00-0.98.hdf5")
    label_dict={0:'Animal', 1:'Architecture', 2:'Scenery', 3:'people', 4:'plane'}
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta()
                  , metrics=["accuracy"])
    img = LoadImage(path)
    res = np.argmax(model.predict(np.array([img])))
    pic_class = label_dict[int(res)]
    print(pic_class)


if __name__ == "__main__":
    path = sys.argv[1]
    predict(path)
