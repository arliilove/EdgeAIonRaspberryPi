'''
树莓派AI实验
目标检测
基于ssd_mobilenet_v2量化模型的Tensorflow Lite模型实现视频'0’，‘1’手势识别
'''
import argparse
import cv2
import numpy as np
import time
from Class import Model, Hand_ROI  # 导入需要的类

'''定义通过命令行读取的相关文件位置/参数'''
parser = argparse.ArgumentParser()
# 存放模型文件夹名(包括.tflite文件和labelmap文件)
parser.add_argument('--modeldir', default='..\TFLite_model\Model_Gesture_1')
# .tflite文件名
parser.add_argument('--graph', default='detect.tflite')
# 标签映射文件的文件名
parser.add_argument('--labels', default='labelmap.txt')
# 目标识别的最低置信度阈值
parser.add_argument('--threshold', default=0.6)
# 视频路径
parser.add_argument('--video', default='Test_videos\hand.MP4')

if __name__ == '__main__':
    '''加载模型类'''
    # 获取命令行中输入的相关文件名及参数
    args = parser.parse_args()
    MODEL_NAME = args.modeldir  # 模型文件夹名
    GRAPH_NAME = args.graph  # .tflite文件名
    LABELMAP_NAME = args.labels  # labelmap文件名
    min_conf_threshold = float(args.threshold)  # 置信度阈值

    # 初始化/加载模型类
    Net = Model()
    Net.load(MODEL_NAME, GRAPH_NAME, LABELMAP_NAME, min_conf_threshold)
    Net.interpreter.allocate_tensors()

    '''准备读入视频'''
    video = cv2.VideoCapture(args.video)
    vw = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    vh = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    _ = video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    '''目标检测'''
    while (True):
        # 获得当前时间点，用于计算 处理一帧图像所需时间（帧速率）
        _, src = video.read()  # 从视频流中读入图片

        '''加载手部ROI类'''
        ROI = Hand_ROI(src)
        frame = ROI.get_roi(ROI.src)  # 利用YCrCb空间获取手部roi
        Net.imH = frame.shape[0]  # 根据roi得到输入图片的长宽
        Net.imW = frame.shape[1]

        '''输入图片(手部roi)预处理'''
        # 输入图片二值化后（保留三通道）转换为RGB
        input_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        # 输入图片调整成网络实际推理时的张量大小
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

        '''可视化结果'''
        if ((scores[i] > Net.min_conf_threshold) and (scores[i] <= 1.0)):  # 只要结果大于最低置信度

            # 标注框坐标有时可能为负值
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = abs(boxes[i][0]), abs(boxes[i][1]), abs(
                boxes[i][2]), abs(boxes[i][3])

            # 有时识别框坐标会超出范围，所以需要与边界比较
            # 加上dx和dy的目的是将roi上识别坐标转化为原图src上的对应坐标

            ymin = int(max(1, (boxes[i][0] * Net.imH) + ROI.dy))
            xmin = int(max(1, (boxes[i][1] * Net.imW) + ROI.dx))
            ymax = int(min(vh, (boxes[i][2] * Net.imH) + ROI.dy))
            xmax = int(min(vw, (boxes[i][3] * Net.imW) + ROI.dx))

            # 框出识别框
            cv2.rectangle(src, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # 画上标签
            object_name = Net.labels[int(classes[i])]  # 获得识别类
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # 输出格式为  类别: XX%
            print("Detect:", label)  # 打印识别结果

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(src, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(src, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)

        # 展示可视化结果
        cv2.imshow('Object Detector', src)

        # 按‘q’键退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 清空窗口
    cv2.destroyAllWindows()
    video.release()
