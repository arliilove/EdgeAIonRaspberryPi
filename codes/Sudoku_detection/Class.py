'''
模型类
Sudoku ROI类
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


class Sudoku_ROI:
    def __init__(self, src, is_showed):
        self.src = src
        self.is_showed = is_showed
        cv2.imshow('Origin', src)  # 展示原图

        self.paper = None
        self.paper_processed = None
        self.sudoku = None
        self.sudoku = None
        self.number_boxes = []
        self.dx_origin = 0
        self.dy_origin = 0

    def find_paper(self, para_paper):
        # 寻找纸张区域
        hsv = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)  # 原bgr图转化为hsv图
        mask = cv2.inRange(hsv, para_paper['white_lower'], para_paper['white_upper'])  # 二值化得到白色区域mask
        mask = cv2.GaussianBlur(mask, para_paper['gauss_kernel'], 0)  # 高斯模糊去除噪点
        if self.is_showed['show_mask']:  # 显示mask
            cv2.imshow('Mask', mask)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 轮廓检测
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对轮廓区域面积进行排序

        '''这里不直接取面积最大的部分的意义在于：背景的白色部分如果与纸是割裂开的，而其他部分的面积一旦大于纸得到的paper便是错误的，所以这种情况下通过被割裂开的纸为矩形判断
        而若纸与背景白色联通，导致若干次后无法单独找出纸，就直接取最大面积的（因为联通，所以一定包含纸）'''

        cnt = 0
        while contours is not None:
            if cnt < len(contours) and cnt < 3:
                contour = contours[cnt]  # 从面积最大的轮廓开始找起
                x, y, w, h = cv2.boundingRect(contour[:, 0, :])  # 保留轮廓的外接矩形左上角顶点和长宽
                if np.abs(((w * h) / cv2.contourArea(contours[0])) - 1.0) <= para_paper['threshold_rec_area']:
                    # 只要外接矩形面积与轮廓面积比值与1.0相差不超过阈值则认为该区域近似矩形，即paper
                    break
            else:
                contourmax = contours[0]  # 直接保留区域面积最大的轮廓作为paper
                x, y, w, h = cv2.boundingRect(contourmax[:, 0, :])  # 保留最大轮廓的外接矩形左上角顶点和长宽
                break

            cnt += 1

        self.paper = self.src[y:y + h, x:x + w]
        self.dx_origin += x
        self.dy_origin += y

        print("Paper found!")
        if self.is_showed['show_paper']:
            # 显示当前认为的的paper区域
            cv2.imshow('Paper', self.paper)

    def paper_preprocess(self, para_preprocess):
        # 预处理：高斯模糊，拉普拉斯算子，膨胀
        paper_preprocessed = cv2.cvtColor(self.paper, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        paper_preprocessed = cv2.GaussianBlur(paper_preprocessed, para_preprocess['gauss_kernel'], 0)  # 高斯模糊
        paper_preprocessed = cv2.Laplacian(paper_preprocessed, cv2.CV_16U,
                                           ksize=para_preprocess['lap_size'])  # 用拉普拉斯算子锐化图像以用于边缘检测
        paper_preprocessed = cv2.convertScaleAbs(paper_preprocessed)  # 图像增强

        kernel = np.ones(para_preprocess['dilate_kernel'], np.uint8)  # 膨胀使边缘加粗
        paper_preprocessed = cv2.dilate(paper_preprocessed, kernel, iterations=para_preprocess['dilate_iter'])

        self.paper_preprocessed = paper_preprocessed
        if self.is_showed['show_paper_preprocessed']:
            # 显示前处理结果
            cv2.imshow('Paper preprocessed', self.paper_preprocessed)

    def find_sudoku(self, para_sudoku):
        # 二值化，取反，腐蚀（通过形态学处理将九宫格部分变为白色正方形区域）
        sudoku = cv2.cvtColor(cv2.cvtColor(self.paper_preprocessed, cv2.COLOR_GRAY2BGR),
                              cv2.COLOR_BGR2HSV)  # 预处理的灰度图转HSV
        sudoku = cv2.inRange(sudoku, para_sudoku['black_lower'], para_sudoku['black_upper'])  # 二值化

        kernel = np.ones(para_sudoku['erode_kernel'], np.uint8)
        sudoku = cv2.erode(sudoku, kernel, iterations=para_sudoku['erode_iter'])  # 腐蚀（使数字与九宫格边框融合）

        sudoku = ~sudoku  # 取反，使sudoku区域变为白色(为了寻找轮廓)
        if self.is_showed['show_square_obtained']:  # 显示
            cv2.imshow('Square_obtained', sudoku)
        contours = cv2.findContours(sudoku, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 得到当前所有联通白色区域外部轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

        #  寻找九宫格：寻找白色正方形sudoku区域
        cnt = 0
        while contours is not None:
            # 从面积最大的轮廓开始找起，只要还没找完并且次数不大于3就继续
            if cnt < len(contours) and cnt < 3:
                contour = contours[cnt]
            else:
                return

            x, y, w, h = cv2.boundingRect(contour[:, 0, :])  # 当前轮廓外接矩形
            if np.abs(w / h - 1.0) <= para_sudoku['threshold_sqr_length']:
                # 是否长宽比接近正方形
                if np.abs(((w * h) / cv2.contourArea(contour)) - 1.0) <= para_sudoku['threshold_sqr_length']:
                    # 是否轮廓面积和外接矩形面积近似
                    self.sudoku = self.paper[y:y + h, x:x + w]
                    self.dx_origin += x
                    self.dy_origin += y
                    print('Sudoku found!')
                    if self.is_showed['show_sudoku']:
                        cv2.imshow('Sudoku', self.sudoku)
                    self.number_split(w, h)
                    return

            cnt += 1

    def number_split(self, w, h):
        # TODO：角点检测
        # gray = cv2.cvtColor(self.sudoku, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray, 2, 1, 0.04)
        # dst = cv2.dilate(dst, None)
        # corners = self.sudoku
        # corners[dst > 0.005 * dst.max()] = [0, 0, 255]
        # cv2.imshow('Coners', corners)

        number = []
        x_step = int((1 / 3) * w)
        y_step = int((1 / 3) * h)
        number.append(self.sudoku[:y_step, :x_step])
        number.append(self.sudoku[:y_step, x_step:2 * x_step])
        number.append(self.sudoku[:y_step, 2 * x_step:])
        number.append(self.sudoku[y_step:2 * y_step, :x_step])
        number.append(self.sudoku[y_step:2 * y_step, x_step:2 * x_step])
        number.append(self.sudoku[y_step:2 * y_step, 2 * x_step:])
        number.append(self.sudoku[2 * y_step:, :x_step])
        number.append(self.sudoku[2 * y_step:, x_step:2 * x_step])
        number.append(self.sudoku[2 * y_step:, 2 * x_step:])
        self.number_boxes = number
