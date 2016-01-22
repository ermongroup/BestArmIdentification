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

            if mean > eps_r:
                return True
            elif mean < -eps_r:
                return False
            r += 1.0

class LILTest:
    def __init__(self, stop_interval=1, a=0.6, c=1.1, b=None):
        self.stop_interval = stop_interval
        self.a = a
        self.c = c
        self.b = b

    def run(self, sim, delta):
        if self.stop_interval == 1:
            a = self.a
            c = self.c
            if self.b is None:
                b = (math.log(zeta(2 * a / c, 1)) - math.log(delta)) * c / 2
            else:
                b = self.b
        else:
            a = self.a
            c = self.stop_interval
            if self.b is None:
                b = (math.log(zeta(2 * a, 1)) - math.log(delta)) / 2
            else:
                b = self.b

        next_stop = 1
        sum = 0.0
        n = 0.0
        while True:
            sum += sim.sim()
            n += 1.0
            boundary = math.sqrt(a * n * math.log(math.log(n, c) + 1) + b * n)
            if self.stop_interval != 1:
                if n < next_stop:
                    continue
                while next_stop < n + 1:
                    next_stop *= c
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
            print("Testing on Bernoulli: mean = " + str(mean)),
            correct_count = 0
            sample_count = np.zeros(self.test_size)

            print_time = time.time()
            for iter in range(0, self.test_size):
                if random.random() > 0.5:
                    mean = -mean
                sim = Simulator(kind="bernoulli", mean=mean)
                result = agent.run(sim, 0.05)
                if (result is True and mean > 0) or (result is False and mean < 0):
                    correct_count += 1
                sample_count[iter] = sim.finish()
                if time.time() - print_time > 1:
                    print(iter),
                    print_time = time.time()

            mean_array[mean_index] = math.fabs(mean)
            accuracy[mean_index] = float(correct_count) / self.test_size
            average_sample[mean_index] = np.sum(sample_count) / self.test_size
            majority_sample[mean_index] = np.sort(sample_count)[int(self.test_size * 0.9)]
            print("")
        return mean_array, accuracy, average_sample, majority_sample

    def compare_and_plot(self, agents, labels, param_list):
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        plt.hold(True)
        #plt.title('Comparison of Required Sample Size')

        counter = 0
        lns = []
        for agent in agents:
            mean, accuracy, average_sample, majority_sample = self.test(agent)
            print(param_list[counter], labels[counter])
            for param, label in zip(param_list[counter], labels[counter]):
                print (param, label)
                if param == 'sample_average':
                    lns += ax2.plot(mean, average_sample, c=color_list[counter], label=label)
                elif param == 'accuracy':
                    lns.append(ax1.scatter(mean, accuracy, c=color_list[counter], label=label))
            counter += 1
        ax1.set_xlim((min(mean), max(mean)))
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax1.set_xlabel('mean')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('number of samples')

        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right')
        plt.show()

    def test_and_plot(self, agent):
        mean, accuracy, average_sample, majority_sample = self.test(agent)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        lns1 = ax1.scatter(mean, accuracy, c='r', label='accuracy')
        lns2 = ax2.plot(mean, average_sample, c='g', label='average sample usage')
        lns3 = ax2.plot(mean, majority_sample, c='b', label='top 10% sample usage')
        ax2.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_ylim((0.8, 1.05))
        ax1.set_xlim((0.01, 1))
        ax2.set_xlim((0.01, 1))
        ax1.set_xlabel('mean')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('samples used')
        lns = [lns1] + lns2 + lns3
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='lower right')
        plt.show()

        return mean, accuracy, average_sample, majority_sample

class SimulationLemma:
    def __init__(self):
        pass

    def run(self, sim, delta):
        k = 1.0
        c = 2 / delta + 1
        while True:
            gamma = math.exp(-k)
            n = 16 / gamma / gamma * math.log(k + c)

            reward_sum = 0.0
            for i in range(0, int(round(n))):
                reward_sum += sim.sim()
            mean = reward_sum / n

            if mean > gamma / 2:
                return True
            elif mean < -gamma / 2:
                return False

            k += 1.0

if __name__ == '__main__':
    sim = Simulator(kind='bernoulli', mean=-0.1)
    exp_gap_agent = ExpGap()
    exp_gap_agent.run(sim, 0.05)
    print(sim.finish())

    lil_agent = LILTest()
    lil_agent.run(sim, 0.05)
    print(sim.finish())

    lil_agent2 = LILTest(stop_interval=1.1)
    lil_agent2.run(sim, 0.05)
    print(sim.finish())

    sl_agent = SimulationLemma()
    sl_agent.run(sim, 0.05)
    print(sim.finish())

    test = TestBench(10)
    #test.test_and_plot(exp_gap_agent)

    #test.compare_and_plot([lil_agent, sl_agent, exp_gap_agent], ['LIL_RW', 'Exponential Gap', 'Simulation Lemma'])

    agent_list = []
    label_list = []
    plot_param = []
    b_list = [3.0, 0.3]
    for b in b_list:
        lil_agent = LILTest(b=b)
        agent_list.append(lil_agent)
        plot_param.append(['accuracy', 'sample_average'])
        label_list.append(["accuracy@b=" + str(b), "average sample@b=" + str(b)])

    test.compare_and_plot(agent_list, label_list, plot_param)

