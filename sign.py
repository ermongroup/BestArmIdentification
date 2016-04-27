__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
from scipy.special import zeta


class Simulator:
    """ Class that simulates a random variable with required mean, and records number of samples requested """
    def __init__(self, kind='bernoulli', mean=0.01):
        self.nsamples = 0
        self.kind = kind
        self.mean = mean

    def sim(self):
        """ Draw a sample from the random variable """
        self.nsamples += 1
        if self.kind == 'bernoulli':
            if random.random() > (self.mean + 1.0) / 2:
                return -0.5
            else:
                return 0.5

    def finish(self):
        """ Clear internal states and return samples collected so far """
        copy = self.nsamples
        self.nsamples = 0
        return copy


class TestBench:
    def __init__(self, size):
        """ size is the number of tests to perform on each configuration to average """
        self.test_size = size

    def test(self, agent):
        """ Run tests on an agent, return list of mean used, accuracy, average samples consumed, and samples
        consumed by top 10% of runs """
        mean_range = 20
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

    def compare_and_plot_sample_count(self, agents, labels):
        """ Run test on multiple agents and compare their average sample complexity """
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        type_list = ['-', '--', '-.', ':']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.hold(True)

        counter = 0
        lns = []
        for agent, label in zip(agents, labels):
            mean, accuracy, average_sample, majority_sample = self.test(agent)
            lns += ax1.plot(mean, average_sample, type_list[counter % len(type_list)],
                            c=color_list[counter % len(color_list)], label=label, lw=4)
            counter += 1

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim((min(mean), max(mean)))

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        ax1.set_xlabel('mean', fontsize=16)
        ax1.set_ylabel('average samples', fontsize=16)
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right', fontsize=16)
        plt.show()

    def compare_and_plot(self, agents, labels, param_list):
        """ Run test on multiple agents and compare their average sample complexity """
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        type_list = ['-', '--', '-.', ':']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        plt.hold(True)

        counter = 0
        lns = []
        use_ax2 = False
        use_ax1 = False
        for agent in agents:
            mean, accuracy, average_sample, majority_sample = self.test(agent)
            print(param_list[counter], labels[counter])
            for param, label in zip(param_list[counter], labels[counter]):
                print (param, label)
                if param == 'sample_average':
                    lns += ax2.plot(mean, average_sample, c=color_list[counter], label=label)
                    use_ax1 = True
                elif param == 'accuracy':
                    lns.append(ax1.scatter(mean, accuracy, c=color_list[counter], label=label))
                    use_ax2 = True
            counter += 1

        ax1.set_xscale('log')

        if use_ax1:
            ax1.set_xlim((min(mean), max(mean)))

            ax1.set_ylabel('accuracy', fontsize=16)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
        if use_ax2:
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_ylabel('number of samples', fontsize=16)
            for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
        ax1.set_xlabel('mean', fontsize=16)

        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right')
        plt.show()

    def test_and_plot(self, agent, ax1, show_legend=False):
        """ Run test on agent and plot results """
        mean, accuracy, average_sample, majority_sample = self.test(agent)

        ax2 = ax1.twinx()

        lns1 = ax1.scatter(mean, accuracy, c='r', label='accuracy')
        lns2 = ax2.plot(mean, average_sample, c='g', label='average sample usage', lw=4)
        lns3 = ax2.plot(mean, majority_sample, c='b', label='top 10% sample usage', lw=4)
        ax2.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_ylim((0.6, 1.05))
        ax1.set_xlim((0.01, 1))
        ax2.set_xlim((0.01, 1))
        ax1.set_xlabel('mean', fontsize=16)
        ax1.set_ylabel('accuracy', fontsize=16)
        ax2.set_ylabel('samples used', fontsize=16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(15)
        ax2.set_ylim((10, 200000))
        lns = [lns1] + lns2 + lns3
        labs = [l.get_label() for l in lns]
        if show_legend:
            plt.legend(lns, labs, loc='lower left', fontsize=16)

        return mean, accuracy, average_sample, majority_sample


class ExpGap:
    """ An agent that uses the exponential gap algorithm """
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
    """ An agent that uses the LIL bound """
    def __init__(self, stop_interval=1, a=0.6, c=1.1, b=None, ratio=1.0):
        self.stop_interval = stop_interval
        self.a = a
        self.c = c
        self.b = b
        self.ratio = ratio

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
            boundary = math.sqrt(a * n * math.log(math.log(n, c) + 1) + b * n) * self.ratio
            if self.stop_interval != 1:
                if n < next_stop:
                    continue
                while next_stop < n + 1:
                    next_stop *= c
            if sum >= boundary:
                return True
            elif sum <= -boundary:
                return False


class SimulationLemma:
    """ An agent that uses constructive proof of second simulation lemma """
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

    lil_agent2 = LILTest(stop_interval=1.05, a=0.55)
    lil_agent2.run(sim, 0.05)
    print(sim.finish())

    sl_agent = SimulationLemma()
    sl_agent.run(sim, 0.05)
    print(sim.finish())

    test = TestBench(100)

    #fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #test.test_and_plot(lil_agent, ax1, show_legend=True)
    #ax1.set_title('Correct Threshold')
    #ax1 = fig.add_subplot(122)
    #test.test_and_plot(lil_agent2, ax1)
    #ax1.set_title('Half Threshold')
    #fig.tight_layout(pad=0.4, w_pad=0)
    #plt.show()

    test.compare_and_plot_sample_count([lil_agent, lil_agent2, sl_agent, exp_gap_agent],
                                       ['AH-RW', 'ESAH-RW', 'Exponential Gap', 'Simulation Lemma'])

    agent_list = []
    label_list = []
    plot_param = []
    b_list = [3.0, 0.3]
    for b in b_list:
        lil_agent = LILTest(b=b)
        agent_list.append(lil_agent)
        plot_param.append(['accuracy', 'sample_average'])
        label_list.append(["accuracy@b=" + str(b), "average sample@b=" + str(b)])

    #test.compare_and_plot(agent_list, label_list, plot_param)

