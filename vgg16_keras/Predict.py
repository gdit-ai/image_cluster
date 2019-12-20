import keras
import cv2
from keras.models import load_model
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def LoadImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize = (64, 64), interpolation = cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    return np.array(img[:, :, :3])

def predict(path):
    abs_path = os.path.abspath(path)
    model = load_model("./vggmodel_21-0.00-0.98.hdf5")
    label_dict={0:'Animal', 1:'Architecture', 2:'Scenery', 3:'people', 4:'plane'}
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta()
                  , metrics=["accuracy"])
    img = LoadImage(abs_path)
    res = np.argmax(model.predict(np.array([img])))
    im = Image.open(abs_path)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('华文圆体 REGULAR.TTF', 30)
    draw.text((30, 10), "识别结果: {}".format(label_dict[int(res)]), fill='#FF0000', font=font)
    im.show()
    print('预测完毕！结果为：', label_dict[int(res)])

if __name__ == "__main__":
    predict('Animal.jpg')
