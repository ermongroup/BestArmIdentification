__author__ = 'shengjia'

from bound_compare import AHBound
import matplotlib.pyplot as plt
import random
import numpy as np
import math


if __name__ == '__main__':
    size = 100000
    index_list = np.zeros(size)
    for i in range(1, size + 1):
        index_list[i-1] = i

    bound = AHBound(0.05).get_bound(index_list)
    incorrect_bound = AHBound(0.05).get_incorrect_bound(index_list)

    cross_count = 0
    incorrect_count = 0
    for i in range(0, 200):
        sum = 0.0
        sum_list = np.zeros(size)
        hit = False
        wrong = False
        for step in range(1, size+1):
            if random.random() > 0.5:
                sum += 0.5
            else:
                sum -= 0.5
            sum_list[step-1] = math.fabs(sum)
            if sum > incorrect_bound[step - 1]:
                hit = True
            if sum > bound[step - 1]:
                wrong = True
        if hit:
            plt.plot(index_list, sum_list, c='b', lw=0.1)
            cross_count += 1
        if wrong:
            incorrect_count += 1

    plt.plot(index_list, bound, c='g', lw=4, label='AH')
    plt.plot(index_list, incorrect_bound, c='r', lw=4, label='Hoeffding')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('n', fontsize=16)
    plt.ylabel('partial sum', fontsize=16)
    for tick in plt.axes().xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in plt.axes().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    print(cross_count, incorrect_count)
    plt.legend(loc='upper left', fontsize=16)
    plt.show()
