__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
from scipy.special import zeta
from numpy.random import normal as nrand

class LILBound:
    def __init__(self, confidence, epsilon=0.01):
        self.confidence = confidence
        self.epsilon = epsilon
        delta = 1.0
        while self.risk(delta) > self.confidence:
            delta /= 1.1
        self.delta = delta

    def risk(self, delta):
        """ return the risk the bound do not hold given delta """
        epsilon = self.epsilon
        rou = 2 * (2.0 + epsilon) / epsilon * ((1 / math.log(1 + epsilon)) ** (1 + epsilon))
        return rou * delta

    def boundary(self, n, delta):
        """ Compute the LIL bound given n, delta """
        epsilon = self.epsilon
        temp = np.log((np.log(1 + epsilon) + np.log(n)) / delta)
        return np.sqrt((1 + epsilon) * temp / 2 * n) * (1 + np.sqrt(epsilon))

    def get_bound(self, index_list):
        bound_list = np.zeros(index_list.size)
        for n, count in zip(index_list, range(0, index_list.size)):
            bound_list[count] = self.boundary(n, self.delta)
        return bound_list

class AHBound:
    def __init__(self, confidence):
        self.confidence = confidence
        self.a = 0.6
        self.c = 1.1
        self.b = (math.log(zeta(2 * self.a / self.c, 1)) - math.log(confidence)) * self.c / 2

    def boundary(self, n):
        return np.sqrt(self.a * n * np.log(np.log(n) / np.log(self.c) + 1)
                       + self.b * n)

    def get_bound(self, index_list):
        bound_list = np.zeros(index_list.size)
        for n, count in zip(index_list, range(0, index_list.size)):
            bound_list[count] = self.boundary(n)
        return bound_list

    def get_incorrect_bound(self, index_list):
        b = -math.log(self.confidence) / 2
        bound_list = np.zeros(index_list.size)
        for n, count in zip(index_list, range(0, index_list.size)):
            bound_list[count] = np.sqrt(b * n)
        return bound_list


class SBBound:
    def __init__(self, confidence):
        self.confidence = confidence

    def boundary(self, n):
        return np.sqrt(3 * n * (2 * np.log(np.log(5 / 2 * n)) + np.log(2 / self.confidence)))

    def get_bound(self, index_list):
        bound_list = np.zeros(index_list.size)
        for n, count in zip(index_list, range(0, index_list.size)):
            bound_list[count] = self.boundary(n)
        return bound_list


class TrivialBound:
    def __init__(self, confidence):
        self.confidence = confidence

    def boundary(self, n):
        return np.sqrt(n / 2 * np.log(2 * n * n / self.confidence))

    def get_bound(self, index_list):
        bound_list = np.zeros(index_list.size)
        for n, count in zip(index_list, range(0, index_list.size)):
            bound_list[count] = self.boundary(n)
        return bound_list


if __name__ == '__main__':

    size = 20
    initial = 5
    confidence_list = [0.01, 0.1]
    c_list = ['r', 'g', 'b', 'm']
    index_list = np.zeros(size)
    for i in range(0, size):
        initial = int(initial * 1.5)
        index_list[i] = initial

    fig = plt.figure()
    for confidence, count in zip(confidence_list, range(0, 2)):
        lil_bound = LILBound(confidence=confidence, epsilon=0.001).get_bound(index_list)
        ah_bound = AHBound(confidence=confidence).get_bound(index_list)
        sb_bound = SBBound(confidence=confidence).get_bound(index_list)
        trivial_bound = TrivialBound(confidence=confidence).get_bound(index_list)
        ax = fig.add_subplot(1, 2, count)
        ax.plot(index_list, ah_bound, '-', c=c_list[0], label=r'AH@$\delta=' + str(confidence) + r'$', lw=4)
        ax.plot(index_list, lil_bound * 1.02, '-.', c=c_list[1], label=r'LIL@$\delta=' + str(confidence) + r'$', lw=4)
        ax.plot(index_list, sb_bound, ':', c=c_list[2], label=r'LIL2@$\delta=' + str(confidence) + r'$', lw=4)
        ax.plot(index_list, trivial_bound, '--', c=c_list[3], label=r'Trivial@$\delta=' + str(confidence) + r'$', lw=4)
        ax.set_xlabel('n', fontsize=20)
        ax.set_ylabel('upper bound', fontsize=20)
        ax.set_ylim((0, 600))
        ax.set_xticks([1000, 5000, 9000, 13000, 17000])
        ax.legend(fontsize=20, loc='lower right')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    plt.show()

