import scipy.io.wavfile
from python_speech_features import *
from universal_utils import write_binary_file
import speech_feature.mfcc as my_mfcc
import numpy as np
import os
import re


def generate_mfcc_samples(**kwargs):
    """
    生成mfcc samples
    :param kwargs: 关键字参数 包括输入文件夹 输出文件夹 输出格式 输出后缀等
    :return: 无
    """
    # 设置初始的关键字参数
    in_dir = kwargs.setdefault("in_dir", "wav")
    in_filter = kwargs.setdefault("in_filter", "\.[Ww][Aa][Vv]")
    out_dir = kwargs.setdefault("out_dir", "mfcc_samples")
    out_ext = kwargs.setdefault("out_ext", ".mfc")
    outfile_format = kwargs.setdefault("outfile_format", "htk")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_list = os.listdir(in_dir)
    for each_file in file_list:
        # 递归遍历 创建文件夹
        if os.path.isdir(os.path.join(in_dir, each_file)):
            kwargs["in_dir"] = os.path.join(in_dir, each_file)
            kwargs["out_dir"] = os.path.join(out_dir, each_file)
            generate_mfcc_samples(**kwargs)
        else:
            # 将音频转化为mfcc
            if re.search(in_filter, each_file):
                infile_name = os.path.join(in_dir, each_file)
                outfile_name = os.path.join(out_dir, each_file.split('.')[0] + out_ext)
                fwav2mfcc(infile_name, outfile_name, outfile_format)


def fwav2mfcc(infile_name, outfile_name, outfile_format, **kwargs):
    """
    音频转mfcc文件
    :param infile_name: 输入文件夹
    :param outfile_name: 输出文件夹
    :param outfile_format: 输出格式
    :param kwargs: 包含mfcc的各种参数
    :return:
    """
    frame_step_sec = kwargs.setdefault("frame_step_sec", 0.01)
    rate, signal = scipy.io.wavfile.read(infile_name)
    # 是否使用库函数
    is_use_lib = kwargs.setdefault(False)
    mfcc_frames = None
    if is_use_lib:
        mfcc_frames = mfcc(signal, rate)
        delta_1 = delta(mfcc_frames, 1)
        delta_2 = delta(mfcc_frames, 1)
        mfcc_frames = np.hstack((mfcc_frames, delta_1, delta_2))
        mfcc_frames = mfcc_frames.T
    else:
        mfcc_frames = my_mfcc.mfcc(signal, rate).T
    if outfile_format.lower() == "htk":
        write_binary_file(outfile_name, mfcc_frames, frame_step_sec)
        print(outfile_name, " Done!")
