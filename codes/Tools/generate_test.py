'''
树莓派AI实验
生成测试集
'''

import os
import numpy as np
import shutil
import argparse

# 从命令行读取相关文件夹路径
CWD_PATH = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--srcDir',default=os.path.join(CWD_PATH, '../images/train/'),help='Path to original dataset')
parser.add_argument('--dstDir',default=os.path.join(CWD_PATH, '../images/test/'),help='Path to test dataset')

def filter(file):  # 用于筛选图片文件
    img_list = []
    for f in file:
        if f[-3:] == 'JPG':
            img_list.append(f)

    return img_list


def generate_test(srcDir, dstDir): # 生成测试集
    data_set = filter(os.listdir(srcDir))  # 从数据集中取出所有.jpg文件

    data_num = len(data_set)  # 数据集图片个数
    test_rate = 0.2  # 自定义测试集占总数据集的比例
    test_num = int(data_num * test_rate)  # 按照rate比例从数据集中取一定数量图片（测试集数量）

    sample = np.random.choice(a=data_set, size=test_num, replace=False)  # 随机选取test_num数量的样本图片
    # 打印选取测试集文件名
    # print(sample)

    for file_name in sample:
        # 将图片移动
        shutil.move(srcDir + file_name, dstDir + file_name)
        # 将图片对应.xml文件移动
        shutil.move(srcDir + file_name[:-3] + 'xml', dstDir + file_name[:-3] + 'xml')

    return


if __name__ == '__main__':
    args = parser.parse_args()
    generate_test(args.srcDir, args.dstDir)  # 生成测试集
    print("test dataset successfully generated")
