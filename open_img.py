import os
import cv2

# 将目录下的所有图片打开，保证所有图片能加载，数据集无误
def openimg(path):
    path = os.path.abspath(path)
    all_file = os.listdir(path)
    all_file.sort()
    print(path)
    for file1 in all_file:
        # print(file1)
        img_file = os.listdir(os.path.join(path, file1))
        img_file.sort()
        for file2 in img_file:
            img = cv2.imread(os.path.join(path, file1, file2))
            print(os.path.join(path, file1, file2))
            # cv2.namedWindow("open", 0)
            # cv2.resizeWindow("open", 400, 200)
            # cv2.moveWindow("open", 0, 0)
            cv2.imshow("open", img)
            # cv2.waitKey(10)
    print('所有图片加载完毕！')
openimg('./data')