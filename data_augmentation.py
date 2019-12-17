from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
dir = './data/'
path = os.path.abspath(dir)
all_file = os.listdir(dir)
all_file.sort()
i = 0
for file1 in all_file:
    # print(file1)
    img_file = os.listdir(os.path.join(path, file1))
    img_file.sort()
    for file2 in img_file:
        img = cv2.imread(os.path.join(path, file1, file2))
        x = img.reshape((1,) + img.shape) # datagen.flow要求rank为4 (1, x.shape)
        datagen.fit(x)
        prefix = file2.split('.')[0] # 以 . 分离字符串，去掉后缀
        counter = 0
        for batch in datagen.flow(x, batch_size=32, save_to_dir=dir + all_file[i], save_prefix=prefix, save_format='jpg'):
            counter += 1
            print(file2, '======', counter)
            if counter > 30:
                break  # 否则生成器会退出循环
    i += 1
