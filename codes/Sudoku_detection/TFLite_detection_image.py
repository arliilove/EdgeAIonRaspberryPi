'''
树莓派AI实验
目标检测
基于ssd_mobilenet_v2量化模型的Tensorflow Lite模型实现单个图像数字九宫格识别
'''
import argparse
import sys
import cv2
import numpy as np
from Class import Model, Sudoku_ROI  # 导入需要的类


def number_detect(input_data, Net):
    '''预处理输入原图片'''
    # 将输入的数字框roi调整成网络实际推理时的张量大小
    src = input_data  # 保留原图
    input_data = cv2.resize(input_data, (Net.input['W'], Net.input['H']))

    # 维度扩充
    input_data = np.expand_dims(input_data, axis=0)

    # 当使用的不是量化模型时，像素归一化
    if Net.input['is_float']:
        input_data = (np.float32(input_data) - Net.input['mean']) / Net.input['std']

    '''模型推理'''
    # 输入数据，推理
    Net.interpreter.set_tensor(Net.input['idx'], input_data)
    Net.interpreter.invoke()

    # 获得模型推理结果
    boxes = Net.interpreter.get_tensor(Net.output['box'])[0]  # 识别框左上角右下角的坐标
    classes = Net.interpreter.get_tensor(Net.output['class'])[0]  # 识别物体的class索引
    scores = Net.interpreter.get_tensor(Net.output['score'])[0]  # 识别置信度

    # 获得最大置信度的识别索引（只找可能性最大的那个）
    i = np.argmax(scores)

    # 获取 类别， 置信度， 矩形框区域
    Class = Net.labels[int(classes[i])]  # Class
    Score = scores[i]  # Score

    boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = abs(boxes[i][0]), abs(boxes[i][1]), abs(
        boxes[i][2]), abs(boxes[i][3])  # 标注框坐标有时可能为负值
    ymin = int(max(0, (boxes[i][0] * Net.imH)))
    xmin = int(max(0, (boxes[i][1] * Net.imW)))
    ymax = int(min(src.shape[0], (boxes[i][2] * Net.imH)))  # 有时识别框坐标会超出范围，所以需要与边界比较
    xmax = int(min(src.shape[1], (boxes[i][3] * Net.imW)))
    Rec = (xmin, ymin, xmax, ymax)  # Rectangle

    return Class, Score, Rec


def visualize(src, results, x_step, y_step, dx_origin, dy_origin, threshold):
    ''' 计算最原始图片中，相对于数字框中坐标的偏移量 dx 和 dy '''
    dx = []
    dy = []
    for wy in [0, 1, 2]:
        for wx in [0, 1, 2]:
            dx.append(dx_origin + wx * x_step)
            dy.append(dy_origin + wy * y_step)

    ''' 在原始图片中标注出所有结果 '''
    cnt = 0
    for result in results:
        if result['Score'] >= threshold:
            # 画出识别框
            xmin, ymin, xmax, ymax = result['Rec'][0] + dx[cnt], result['Rec'][1] + dy[cnt], result['Rec'][2] + dx[cnt], \
                                     result['Rec'][3] + dy[cnt]
            cv2.rectangle(src, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # 画出标签
            object_name = result['Class']  # 获得识别类
            label = '%s %d%%' % (object_name, int(result['Score'] * 100))  # 输出格式为  类别: XX%
            print("Detect:", label)  # 打印识别结果

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(src, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(src, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)
        cnt += 1
    cv2.imshow('Result', src)  # 显示结果


'''定义通过命令行读取的相关文件位置/参数'''
parser = argparse.ArgumentParser()
# 存放模型文件夹名(包括.tflite文件和labelmap文件)
parser.add_argument('--modeldir', default='../TFLite_model/Model_Sudoku_2')
# .tflite文件名
parser.add_argument('--graph', default='detect.tflite')
# 标签映射文件的文件名
parser.add_argument('--labels', default='labelmap.txt')
# 目标识别的最低置信度阈值
parser.add_argument('--threshold', default=0.2)
# 图片路径
parser.add_argument('--image', default='Test_images/Complete/sudoku_in_white_wall.JPG')

# 获取命令行中输入的模型相关文件名及参数
args = parser.parse_args()
MODEL_NAME = args.modeldir  # 模型文件夹名
GRAPH_NAME = args.graph  # .tflite文件名
LABELMAP_NAME = args.labels  # labelmap文件名
min_conf_threshold = float(args.threshold)  # 置信度阈值

''' 设定相关参数 '''
is_showed = {'show_mask': True, 'show_paper': True, 'show_paper_preprocessed': True, 'show_square_obtained': True,
             'show_sudoku': True}
para_paper = {'white_lower': np.array([0, 0, 0]), 'white_upper': np.array([180, 60, 255]), 'gauss_kernel': (5, 5),
              'threshold_rec_area': 0.3}
para_preprocess = {'gauss_kernel': (3, 3), 'lap_size': 3, 'dilate_kernel': (4, 4), 'dilate_iter': 2}
para_sudoku = {'black_lower': np.array([0, 0, 0]), 'black_upper': np.array([180, 255, 55]), 'erode_kernel': (5, 5),
               'erode_iter': 4, 'threshold_sqr_length': 0.3, 'threshold_sqr_area': 0.3}

if __name__ == '__main__':
    '''读入图片'''
    src = cv2.imread(args.image)

    '''加载Sudoku ROI类并获取Sudoku_ROI'''
    ROI = Sudoku_ROI(src, is_showed)  # 生成ROI类实例
    # 首先框定纸区域
    ROI.find_paper(para_paper)
    if ROI.paper is None:
        print('Paper not found!')
        sys.exit()

    # 寻找九宫格
    ROI.paper_preprocess(para_preprocess)  # 预处理
    ROI.find_sudoku(para_sudoku)  # 寻找九宫格, 并分割九数字
    if ROI.sudoku is None:
        print('Sudoku not found! Closer!')
        sys.exit()

    ''' 检测，获得结果 '''
    results = []  # 存储识别结果
    cnt = 0
    for input_data in ROI.number_boxes:  # 逐个检测每一个数字框
        cnt += 1
        # 初始化/加载模型类
        Net = Model()
        Net.load(MODEL_NAME, GRAPH_NAME, LABELMAP_NAME, min_conf_threshold)
        Net.interpreter.allocate_tensors()
        Net.imH = input_data.shape[0]  # 根据roi得到输入图片的长宽
        Net.imW = input_data.shape[1]

        # 目标检测
        Class, Score, Rec = number_detect(input_data, Net)
        results.append({'Class': Class, 'Score': Score, 'Rec': Rec})

    """ 可视化结果 """
    visualize(src, results, x_step=int((1 / 3) * ROI.sudoku.shape[1]), y_step=int((1 / 3) * ROI.sudoku.shape[0]),
              dx_origin=ROI.dx_origin, dy_origin=ROI.dy_origin, threshold=Net.min_conf_threshold)

    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 清空窗口
