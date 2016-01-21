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
                return -1
            else:
                return 1


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

class LilTest:
    def __init__(self):
        pass

    def run(self, sim, delta):
        a = 0.8
        c = 1.1
        b = (math.log(zeta(2 * a / c, 1)) - math.log(delta)) * c / 2
        print(a, b, c)

        sum = 0.0
        n = 0.0
        while True:
            sum += sim.sim()
            n += 1.0
            boundary = math.sqrt(a * n * math.log(math.log(n, c) + 1) + b * n)
            print(sum, boundary)

            if sum >= boundary:
                return True
            elif sum <= -boundary:
                return False


if __name__ == '__main__':
    sim = Simulator(kind='bernoulli', mean=-0.1)
    exp_gap_agent = ExpGap()
    exp_gap_agent.run(sim, 0.1)