import csv
import numpy as np


def get_data(filename):
    """
    根据filename读取所对应的音频文件位置及对应的类别
    :param filename: 训练数据csv或测试数据csv文件名
    :return: 从csv中读取一个列表，每一个列表的元素也是一个列表，有类别和文件路径组成
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([int(row[0]), row[1]])
    return data


def generate_training_list():
    """
    生成训练数据csv文件，每一行包括两列，第一列为所属类别，第二列为文件路径
    :return: 无
    """
    list_filename = "training_file.csv"
    MODEL_NO = 11
    dir1 = "mfcc_samples"
    dir3 = ['AE', 'AJ', 'AL', 'AW', 'BD', 'CB', 'CF', 'CR', 'DL', 'DN', 'EH', 'EL', 'FC', 'FD', 'FF', 'FI', 'FJ', 'FK',
            'FL', 'GG']
    wordids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    list_file = []
    for d3 in dir3:
        for k in range(1, len(wordids) + 1):
            for p in ['A', 'B']:
                list_file.append([k, '%s/%s/%c%c_endpt.mfc' % (dir1, d3, wordids[k - 1], p)])
    with open(list_filename, 'w') as file:
        for t in list_file:
            file.write("%s,%s\n" % (t[0], t[1]))


def generate_testing_list():
    """
    生成测试数据csv文件，每一行包括两列，第一列为所属类别，第二列为文件路径
    :return: 无
    """
    list_filename = "testing_file.csv"
    dir1 = "mfcc_samples"
    dir3 = ['AH', 'AR', 'AT', 'BC', 'BE', 'BM', 'BN', 'CC', 'CE', 'CP', 'DF', 'DJ', 'ED', 'EF', 'ET', 'FA', 'FG', 'FH',
            'FM', 'FP', 'FR', 'FS', 'FT', 'GA', 'GP', 'GS', 'GW', 'HC', 'HJ', 'HM', 'HR', 'IA', 'IB', 'IM', 'IP', 'JA',
            'JH', 'KA', 'KE', 'KG', 'LE', 'LG', 'MI', 'NL', 'NP', 'NT', 'PC', 'PG', 'PH', 'PR', 'RK', 'SA', 'SL', 'SR',
            'SW', 'TC']
    word_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    list_file = []
    for d3 in dir3:
        for k in range(1, len(word_ids) + 1):
            for p in ['A', 'B']:
                list_file.append([k, '%s/%s/%c%c_endpt.mfc' % (dir1, d3, word_ids[k - 1], p)])
    with open(list_filename, 'w') as file:
        for t in list_file:
            file.write("%s,%s\n" % (t[0], t[1]))


def write_binary_file(filename, feature, frame_step_sec):
    """
    以二进制格式把mfcc写入文件中
    :param filename: 文件名
    :param feature: mfcc帧
    :param frame_step_sec: 这个在matlab代码中并没有用到 但为保持一致性 将其保留
    :return:
    """
    dim, frame_no = feature.shape
    with open(filename, 'wb') as file:
        file.write(frame_no.to_bytes(length=4, byteorder='big', signed=True))
        sample_period = round(frame_step_sec * 1E7)
        file.write(sample_period.to_bytes(length=4, byteorder='big', signed=True))
        sample_size = dim * 4
        file.write(sample_size.to_bytes(length=2, byteorder='big', signed=True))
        parm_kind = 838
        file.write(parm_kind.to_bytes(length=2, byteorder='big', signed=True))
        file.write(feature.tobytes())


def read_binary_file(filename):
    """
    以二进制格式从文件中读取mfcc
    :param filename: 文件名
    :return: speech_feature
    """
    with open(filename, 'rb') as file:
        n_samples = int.from_bytes(file.read(4), byteorder='big', signed=True)
        int.from_bytes(file.read(4), byteorder='big', signed=True) * 1E-7
        sample_size = int.from_bytes(file.read(2), byteorder='big', signed=True)
        dim = int(0.25 * sample_size)
        int.from_bytes(file.read(2), byteorder='big', signed=True)
        data = file.read()
        feature = np.array(np.frombuffer(data, dtype=np.float64))
        feature = feature.reshape((dim, n_samples))
        return feature
