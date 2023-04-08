import numpy as np
import math


def pre_emphasis(signal, alpha=0.97):
    """ pre-emphasis before handle the input signal
    :param signal: 输入信号
    :param alpha: 默认0.97 通常为0.96~0.98
    :return: 预加重后的信号
    """
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal


def windowing(signal, rate, frame_size_sec=0.025, frame_step_sec=0.01, window_name="hamming"):
    """frame a signal into overlapping frames
    :param signal: 输入信号
    :param rate: 采样率
    :param frame_size_sec: 窗口分析时长
    :param frame_step_sec: 窗口增长时长
    :param window_name: 加窗函数名
    :return: 加窗后的信号帧
    """
    # Popularly set a frame size of 25 ms and 10 ms step (15 ms overlap).
    frame_size = int(frame_size_sec * rate)
    frame_step = int(frame_step_sec * rate)
    signal_length = len(signal)
    frame_nums = math.ceil((signal_length - frame_size) / frame_step)
    # padding 0 when a window is not enough for a frame size
    padding_length = frame_nums * frame_step + frame_size
    padding = np.zeros(padding_length - signal_length)
    signal_padding = np.append(signal, padding)
    # windowing index
    index = np.array([[j + i * frame_step for j in range(frame_size)] for i in range(0, frame_nums)])
    window = None
    if window_name == "hamming":
        window = np.hamming(frame_size)
    elif window_name == "blackman":
        window = np.blackman(frame_size)
    elif window_name == "bartlett":
        window = np.bartlett(frame_size)
    else:
        raise ValueError("There is no %s." % window_name)
    windowing_frames = signal_padding[index]
    windowing_frames *= window
    return windowing_frames


def dft(signal, nfft):
    """ fft transform
    :param signal: 输入信号
    :param nfft: n维变换
    :return: 傅里叶变换后的帧
    """
    STFT_frames = np.abs(np.fft.rfft(signal, nfft))
    pow_frames = (STFT_frames ** 2) / nfft
    return pow_frames


def mel_filter(nfft, pow_frames, rate, filter_banks=26):
    """
    :param nfft: n维变换
    :param pow_frames: 平方后的帧
    :param rate: 采样率
    :param filter_banks: 滤波器组
    :return: 特征
    """
    # frequency:[0, sample_rate / 2]
    low_mel = 0
    high_mel = 1127 * np.log(1 + rate / 2 / 700)
    # calculate mel points
    mel_points = np.linspace(low_mel, high_mel, filter_banks + 2)
    # convert thess back to Hertz
    freq_points = 700 * (np.exp(mel_points / 1127) - 1)
    # the nearest FFT bin
    bin_points = np.floor((nfft + 1) * freq_points / rate).astype(np.int32)
    # filter_banks
    feature = np.zeros((filter_banks, int(np.floor(nfft / 2 + 1))))
    for m in range(1, filter_banks + 1):
        f_min = bin_points[m - 1]
        f_mid = bin_points[m]
        f_max = bin_points[m + 1]
        feature[m - 1, f_min: f_mid] = (np.arange(f_min, f_mid) - bin_points[m - 1]) / (
                bin_points[m] - bin_points[m - 1])
        feature[m - 1, f_mid: f_max] = (bin_points[m + 1] - np.arange(f_mid, f_max)) / (
                bin_points[m + 1] - bin_points[m])
    feature = np.dot(pow_frames, feature.T)
    return feature
