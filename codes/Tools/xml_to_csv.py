'''
树莓派AI实验
.xml转换为.csv文件
'''
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import numpy as np

# 从命令行读取相关文件夹路径
CWD_PATH = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default=os.path.join(CWD_PATH, '../images/'),help='Path to the image directory')

def xml_to_csv(path,folder_name):
    csv_list = []  # 用于记录生成dataframe的属性列表
    files = glob.glob(path + '/*.xml')
    np.random.shuffle(files)

    for xml_file in files:  # 获得当前voc数据集中所有xml文件名集合
        tree = ET.parse(xml_file)  # 将xml文件转化为树
        root = tree.getroot()  # 获取根节点
        # 这里因为对于纯负样本集没有相关参数，所以添加异常

        for member in root.findall('object'):  # 获得object标签下所有子标签内容
            '''
            #用value元组记录构成dataframe必要的属性内容
            #    图片文件名
            #    大小（长和宽）
            #    object的class名称
            #    object的xmin, ymin, xmax, ymax
            '''
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            csv_list.append(value)  # 更新csv_list

    column_label = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']  # csv列标签
    csv_df = pd.DataFrame(csv_list, columns=column_label)  # 生成用于生成csv文件的dataframe
    csv_path = '../images/' + folder_name + '.csv'  # 生成csv文件路径
    csv_df.to_csv(csv_path, index=None)  # 转化csv文件

    return

if __name__ == '__main__':
    args = parser.parse_args()
    for folder_name in ['train','test']:  # 训练集、测试集分别生成csv文件
        data_set = os.path.join(args.img_dir, folder_name)  # 当前数据集路径
        xml_to_csv(data_set, folder_name)  # 从数据集图片中生成dataframe

    print('.xml successfully converted to .csv')
