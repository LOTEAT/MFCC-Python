import os.path

import matplotlib.pyplot as plt
from hmm import HMM
import numpy as np
import argparse
from mfcc_utils import generate_mfcc_samples
import time

# 消除numpy的所有警告
np.seterr(all='ignore')
# 创建句柄
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--num_of_state_start', '-start', type=int, default=12)
parser.add_argument('--num_of_state_end', '-end', type=int, default=15)
parser.add_argument('--dim', '-dim', type=int, default=39)
parser.add_argument('--num_of_model', '-model', type=int, default=11)
parser.add_argument('--flag', '-flag', type=str, default="true")
parser.add_argument('--lib_flag', '-lib_flag', type=str, default="true")


def main():
    args = parser.parse_args()

    num_of_state_start = args.num_of_state_start
    num_of_state_end = args.num_of_state_end
    DIM = args.dim
    num_of_model = args.num_of_model
    flag = args.flag
    lib_flag = args.lib_flag
    accuracy_rate = []

    # 是否需要调用库函数
    if lib_flag.lower() == "false":
        lib_flag = False
    elif lib_flag.lower() == "true":
        lib_flag = True
    else:
        raise ValueError("There is no %s." % lib_flag)

    # 是否需要重新生成mfcc samples
    if flag.lower() == "false":
        pass
    elif flag.lower() == "true":
        time_start = time.time()
        generate_mfcc_samples(is_use_lib=lib_flag)
        print('\n')
        if lib_flag:
            print("采用mfcc库函数生成mfcc sample")
        else:
            print("采用自己实现的mfcc生成mfcc sample")
        time_end = time.time()
        print('time cost:', time_end - time_start, 's')
    else:
        raise ValueError("There is no %s." % flag)
    for num_of_state in range(num_of_state_start, num_of_state_end + 1):
        print("\nnum_of_state: %d " % num_of_state)
        hmm = HMM(num_of_model=num_of_model, num_of_state=num_of_state)
        hmm.train_model()
        accuracy = hmm.test_model()
        accuracy_rate.append(accuracy)
    plt.figure()
    plt.plot(accuracy_rate)
    plt.savefig(os.path.join("imgs", "ACC") + ".png")
    plt.show()


if __name__ == '__main__':
    main()
