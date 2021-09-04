'''
模型类
手部ROI类
'''

import os
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import cv2

'''模型类'''
class Model:
    def __init__(self):
        self.interpreter = None  # 用于推理的神经网络本体
        self.min_conf_threshold = None  # 模型输出置信度阈值
        self.labels = None  # 模型标签集
        self.imW = None  # 网络输入图片的长
        self.imH = None  # 网络输入图片的宽

        '''
        self.input: 网络输入的参数
        'H','W': 网络推理张量的长宽
        'is_float': 是否为浮点型（非量化模型）
        如果模型非量化模型时像素的均值和标准差，即255的一半(用于归一化，参考灰度世界算法)
        网络输入的index
        '''
        self.input = {'W': None, 'H': None, 'is_float': None, 'mean': 127.5, 'std': 127.5, 'idx': None}

        '''
        网络输出的参数:
        网络输出 框/类/置信度 的index
        '''
        self.output = {'box': None, 'class': None, 'score': None}

    def load(self, MODEL_NAME, GRAPH_NAME, LABELMAP_NAME, min_conf_threshold):  # 加载模型
        # 获取相关路径
        CWD_PATH = os.getcwd()  # 获取当前位置
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)  # 获取模型位置
        PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)  # 获取labelmap位置

        # 加载标签labels
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        if self.labels[0] == '???':  # 如果使用官方coco数据集测试，第一个label为???，要去掉
            del (self.labels[0])

        # 设置最低置信度阈值
        self.min_conf_threshold = min_conf_threshold

        # 加载模型interpreter
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        # self.interpreter.allocate_tensors()

        # 获取模型输入（张量）、输出相关参数
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input['H'] = input_details[0]['shape'][1]  # 网络实际推理时输入的张量长宽
        self.input['W'] = input_details[0]['shape'][2]
        self.input['is_float'] = (input_details[0]['dtype'] == np.float32)  # 张量类型是否为浮点型（是否为量化模型）
        self.input['idx'] = input_details[0]['index']  # 输入索引

        self.output['box'] = output_details[0]['index']  # 输出boxes索引
        self.output['class'] = output_details[1]['index']  # 输出classes索引
        self.output['score'] = output_details[2]['index']  # 输出scores索引

'''手部ROI类'''
class Hand_ROI:
    def __init__(self, src, roi=None, dx=None, dy=None):
        self.src = src  # 原图像
        self.roi = roi  # 手部兴趣区域，即神经网络的输入
        self.dx = dx  # dx,dy即ROI在原图中的坐标，用于将识别结果的坐标转换为原图坐标
        self.dy = dy

    def skin_detect(self, img):  # 肤色检测
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # 将 src转换至YCrCb空间
        (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
        cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)  # 对Cr空间高斯模糊
        thresh, skin_mask = cv2.threshold(cr_blur, 0, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 用Ostu二值化,thresh为聚类得到阈值，skin_mask为二值化后图像
        res = cv2.bitwise_and(img, img, mask=skin_mask)  # 掩膜得到肤色检测结果

        # 展示肤色检测结果
        # cv2.imshow('Skin_Area', res)

        return res

    def find_hand(self, img):  # 寻找手部
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 寻找每个轮廓，存储构成每个轮廓的点集
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对轮廓区域面积进行排序
        contourmax = contours[0][:, 0, :]  # 保留区域面积最大的轮廓点坐标

        x, y, w, h = cv2.boundingRect(contourmax)  # 外接矩形左上角顶点和长宽
        return x, y, w, h

    def get_roi(self, src):
        Skin_Area = self.skin_detect(src)  # YCrCb空间中肤色检测结果

        Skin_Area = cv2.cvtColor(Skin_Area, cv2.COLOR_BGR2GRAY)  # 将检测结果转化为灰度图
        Skin_Area = cv2.GaussianBlur(Skin_Area, (5, 5), 0)  # 对肤色区域高斯模糊
        Skin_Area = cv2.Laplacian(Skin_Area, cv2.CV_16S, ksize=3)  # 用拉普拉斯算子锐化图像以用于边缘检测
        Skin_Area = cv2.convertScaleAbs(Skin_Area)  # 图像增强

        x, y, w, h = self.find_hand(Skin_Area)  # 运用轮廓检测..寻找手部区域

        self.dx = x  # 记录roi位置
        self.dy = y
        self.roi = src[y:y + h, x:x + w]  # 在原图上截出roi区域存储

        # 实时展示roi
        # cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)
        # cv2.imshow('Hand_ROI', src)

        return self.roi