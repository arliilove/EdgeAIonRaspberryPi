import numpy as np
import cv2
import os

for folder_name in ['test/','train/']:
    dir_path = os.path.join(os.getcwd(),('../images/'+folder_name))
    for filename in os.listdir(dir_path):
        filename = os.path.join(dir_path,filename)
        if filename.endswith(".JPG"):
            image = cv2.imread(filename)
            # 将图片按比例缩小（根据需求调整）
            resized = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename,resized)

print("images successfully resized")
