'''
树莓派AI实验
从csv生成tfrecord文件
'''
import os
import pandas as pd
from PIL import Image
from tensorflow.python.framework.versions import VERSION

# 确保导入的是tensorflow 1.x版本的
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
import argparse

CWD_PATH = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default=os.path.join(CWD_PATH, '../images/'), help='Path to the image directory')
parser.add_argument('--csv_dir', default=os.path.join(CWD_PATH, '../images/'), help='Path to the input the csv')
parser.add_argument('--output_dir', default=os.path.join(CWD_PATH, '../training/'),
                    help='Path to the output the TFRecord')

# TODO: 将class改成自己数据集对应要识别的类
def class_text_to_label(row_label):
    if row_label == 'one':
        return 1
    elif row_label == 'two':
        return 2
    elif row_label == 'three':
        return 3
    elif row_label == 'four':
        return 4
    elif row_label == 'five':
        return 5
    elif row_label == 'six':
        return 6
    elif row_label == 'seven':
        return 7
    elif row_label == 'eight':
        return 8
    elif row_label == 'nine':
        return 9
    else:
        pass


def create_tf_example(example, img_path):
    # 读取对应原始图片（bytes）
    path = os.path.join(img_path, '{}'.format(example['filename']))  # 图像路径
    encoded_jpg = tf.gfile.GFile(path, 'rb').read()  # 原始图像
    # 图片长宽`
    img = Image.open(path)
    width, height = img.size

    # 图像格式
    image_format = b'jpg'

    # 文件名
    filename = example['filename'].encode('utf8')

    # box框的xmin,xmax,ymin,ymax
    xmin = example['xmin'] / width
    xmax = example['xmax'] / width
    ymin = example['ymin'] / width
    ymax = example['ymax'] / width

    # 类的名称，标签（数字）
    # 注意，如果标签为数字默认类型不是字符串而是np.int64
    class_text = example['class'].encode('utf8')
    class_label = class_text_to_label(example['class'])

    # 生成Example协议块
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[xmin])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[xmax])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[ymin])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[ymax])),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_text])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_label])),
    }))
    return tf_example



if __name__ == '__main__':
    args = parser.parse_args()
    for folder_name in ['train', 'test']:
        # 读取相应文件路径
        img_path = os.path.join(args.img_dir, (folder_name + '/'))
        csv_path = os.path.join(args.csv_dir, (folder_name + '.csv'))
        output_path = os.path.join(args.output_dir, (folder_name + '.record'))

        # 建立TFRecord存储器
        writer = tf.python_io.TFRecordWriter(output_path)

        # 读取当前csv文件中所有样本
        examples = pd.read_csv(csv_path)
        for i in range(len(examples['filename'])):
            tf_example = create_tf_example(examples.loc[i], img_path)  # 构造每个样本的Example协议块
            writer.write(tf_example.SerializeToString())

        writer.close()

        print('the TFRecord created successfully: {}'.format(folder_name + '.record'))
