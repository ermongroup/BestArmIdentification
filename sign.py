__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
from scipy.special import zeta


class Simulator:
    def __init__(self, kind='bernoulli', mean=0.01):
        self.nsamples = 0
        self.kind = kind
        self.mean = mean

    def sim(self):
        self.nsamples += 1
        if self.kind == 'bernoulli':
            if random.random() > (self.mean + 1.0) / 2:
                return -0.5
            else:
                return 0.5

    def finish(self):
        copy = self.nsamples
        self.nsamples = 0
        return copy

class ExpGap:
    def __init__(self):
        pass

    def run(self, sim, delta):
        r = 1.0
        while True:
            eps_r = math.exp(-r) / 2
            delta_r = delta / 10 / r / r
            t_r = 2 * math.log(2 / delta_r) / eps_r / eps_r

            reward_sum = 0.0
            for i in range(0, int(round(t_r))):
                reward_sum += sim.sim()
            mean = reward_sum / t_r

            print("mean=" + str(mean) + ", eps_r = " + str(eps_r) + ", delta_r = " + str(delta_r) + ", t_r = " + str(t_r))
            if mean > eps_r:
                return True
            elif mean < -eps_r:
                return False
            r += 1.0

class LILTest:
    def __init__(self):
        pass

    def run(self, sim, delta):
        a = 0.6
        c = 1.1
        b = (math.log(zeta(2 * a / c, 1)) - math.log(delta)) * c / 2
        print(a, b, c)

        sum = 0.0
        n = 0.0
        while True:
            sum += sim.sim()
            n += 1.0
            boundary = math.sqrt(a * n * math.log(math.log(n, c) + 1) + b * n)
            if sum >= boundary:
                return True
            elif sum <= -boundary:
                return False

class TestBench:
    def __init__(self, size):
        self.test_size = size

    def test(self, agent):
        mean_range = 22
        mean_array = np.zeros(mean_range)
        accuracy = np.zeros(mean_range)
        average_sample = np.zeros(mean_range)
        majority_sample = np.zeros(mean_range)
        for mean_index in range(0, mean_range):
            mean = math.exp(-0.2 * mean_index)
            print("Testing on Bernoulli: mean = " + str(mean))
            correct_count = 0
            sample_count = np.zeros(self.test_size)
            for iter in range(0, self.test_size):
                if random.random() > 0.5:
                    mean = -mean
                sim = Simulator(kind="bernoulli", mean=mean)
                result = agent.run(sim, 0.05)
                if (result is True and mean > 0) or (result is False and mean < 0):
                    correct_count += 1
                sample_count[iter] = sim.finish()

            mean_array[mean_index] = math.fabs(mean)
            accuracy[mean_index] = float(correct_count) / self.test_size
            average_sample[mean_index] = np.sum(sample_count) / self.test_size
            majority_sample[mean_index] = np.sort(sample_count)[int(self.test_size * 0.9)]
        return mean_array, accuracy, average_sample, majority_sample

if __name__ == '__main__':
    sim = Simulator(kind='bernoulli', mean=-0.1)
    exp_gap_agent = ExpGap()
    exp_gap_agent.run(sim, 0.1)
    print(sim.finish())

    lil_agent = LILTest()
    lil_agent.run(sim, 0.1)
    print(sim.finish())

    test = TestBench(200)
    mean, accuracy, average_sample, majority_sample = test.test(lil_agent)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.title('Accuracy and Required Samples vs. Mean')
    lns1 = ax1.scatter(mean, accuracy, c='r', label='accuracy')
    lns2 = ax2.plot(mean, average_sample, c='g', label='average sample usage')
    lns3 = ax2.plot(mean, majority_sample, c='b', label='top 10% sample usage')
    ax2.set_yscale('log')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlim((0.01, 1))
    ax2.set_xlim((0.01, 1))
    ax1.set_xlabel('mean')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('samples used')
    lns = [lns1] + lns2 + lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='lower right')
    plt.show()

