from scipy.fftpack import dct
from speech_feature.process import *


def mfcc(signal, rate, **kwargs):
    """
    计算mfcc
    :param signal: 原始信号
    :param rate: 采样率
    :param kwargs: 关键字参数 包括特征提取数 倒谱升降器数量 差分阶数等等
                通过向该关键字参数赋值 可以获得不同的mfcc
    :return: mfcc
    """
    num_cep = kwargs.setdefault("num_cep", 13)
    cep_lifter = kwargs.setdefault("cep_lifter", 22)
    delta = kwargs.setdefault("delta", 3)
    feature, energy = signal_filter(signal, rate, **kwargs)
    feature = dct(feature, type=2, axis=1, norm='ortho')[:, :num_cep]
    mfcc_frames = dfe(feature, energy, cep_lifter)
    for i in range(1, delta):
        feature_delta = delta_feature(feature, i)
        mfcc_frames = np.hstack((mfcc_frames, feature_delta))
    return mfcc_frames


def signal_filter(signal, rate, **kwargs):
    """
    信号滤波
    :param signal: 原始信号
    :param rate: 采样率
    :param kwargs: 关键字参数 包括alpha 窗口长度 窗口每次增长长度 滤波器组数等
    :return: 滤波后特征和能量
    """
    alpha = kwargs.setdefault("alpha", 0.97)
    frame_size_sec = kwargs.setdefault("frame_size_sec", 0.025)
    frame_step_sec = kwargs.setdefault("frame_step_sec", 0.01)
    window_name = kwargs.setdefault("window", "hamming")
    filter_banks = kwargs.setdefault("filter_banks", 26)

    # calculate nfft
    nfft = 1
    while nfft < frame_size_sec * rate:
        nfft *= 2

    pre_emphasis_signal = pre_emphasis(signal, alpha)
    window_frames = windowing(pre_emphasis_signal, rate, frame_size_sec, frame_step_sec, window_name)
    pow_frames = dft(window_frames, nfft)
    energy = np.sum(pow_frames, 1)
    feature = mel_filter(nfft, pow_frames, rate, filter_banks)
    # log(0) may cause some problems
    feature = np.where(feature == 0, np.finfo(float).eps, feature)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feature = np.log(feature)
    energy = np.log(energy)
    return feature, energy


def dfe(feature, energy, cep_lifter=22):
    """
    dynamic feature extraction
    :param feature: 特征
    :param energy: 能量
    :param cep_lifter: 提升器
    :return: 提取后的特征（能量替换值第一维）
    """
    _, cols = feature.shape
    n = np.arange(cols)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    feature *= lift
    # for the features of the 0th dimension are quite different
    # from the features of other dimensions, so we replace it
    # with energy
    feature[:, 0] = energy
    return feature


def delta_feature(feature, n):
    """
    求mfcc的差分
    :param feature: mfcc特征
    :param n: 差分阶数
    :return: n阶差分后的mfcc特征
    """
    feature_length = feature.shape[0]
    s = 2 * sum([i ** 2 for i in range(1, n + 1)])
    delta_feature = np.zeros_like(feature)
    # padding, edge mode
    feature_padding = np.pad(feature, ((n, n), (0, 0)), mode='edge')
    for t in range(feature_length):
        delta_feature[t] = np.dot(np.arange(-n, n + 1), feature_padding[t: t + 2 * n + 1]) / s
    return delta_feature
