__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time

class Simulator:
    def __init__(self, K, kind='H2'):
        self.K = K
        self.nsamples = 0
        self.kind = kind
        self.sample_size = np.zeros(K)

        self.mean_list = np.zeros(K)
        for arm in range(0, K):
            if self.kind == 'H0':
                if arm == 0:
                    self.mean_list[arm] = 0.6
                else:
                    self.mean_list[arm] = 0.3
            elif self.kind == 'H2':
                self.mean_list[arm] = 0.9 - (float(arm) / self.K) ** 0.6
                if self.mean_list[arm] < 0:
                    self.mean_list[arm] = 0
            elif self.kind == 'H':
                self.mean_list[arm] = 0.5

    def sim(self, arm):
        self.nsamples += 1
        self.sample_size[arm] += 1
        if random.random() > self.mean_list[arm]:
            return 0
        else:
            return 1

    def hardness(self):
        hardness_sum = 0.0
        for arm in range(1, self.K):
            hardness_sum += 1.0 / ((self.mean_list[0] - self.mean_list[arm]) ** 2)
        return hardness_sum

    def finish(self):
        copy = self.nsamples
        self.nsamples = 0
        return copy

    def plot_samples(self):
        plt.clf()
        plt.scatter(range(0, self.K), self.sample_size)
        plt.yscale('log')
        plt.ioff()
        plt.show()

class MABTestBench:
    def __init__(self, kind, size):
        self.kind = kind
        self.size = size
        self.k_range = 5

    def test(self, agent):
        narms_array = np.zeros(self.k_range)
        accuracy = np.zeros(self.k_range)
        average_sample = np.zeros(self.k_range)
        majority_sample = np.zeros(self.k_range)
        K = 2
        for i in range(0, self.k_range):
            print("Testing on K=" + str(K))
            correct_count = 0
            sample_count = np.zeros(self.size)
            for rep in range(0, self.size):
                print("    " + str(rep) + "-th iteration")
                sim = Simulator(K, self.kind)
                if agent.run(sim) == 0:
                    correct_count += 1
                sample_count[rep] = sim.finish()

            narms_array[i] = K
            accuracy[i] = float(correct_count) / self.size
            average_sample[i] = np.sum(sample_count) / self.size / sim.hardness()
            majority_sample[i] = np.sort(sample_count)[int(self.size * 0.9)] / sim.hardness()
            K *= 2
        return narms_array, accuracy, average_sample, majority_sample

    def test_and_plot(self, agent):
        n_arms, accuracy, average_sample, majority_sample = self.test(agent)
        print(n_arms, accuracy, average_sample, majority_sample)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        lns1 = ax1.scatter(n_arms, accuracy, c='r', label='accuracy')
        lns2 = ax2.plot(n_arms, average_sample, c='g', label='average sample usage')
        lns3 = ax2.plot(n_arms, majority_sample, c='b', label='top 10% sample usage')
        ax2.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_ylim((0.8, 1.05))
        ax1.set_xlabel('number of arms')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('samples used')
        lns = [lns1] + lns2 + lns3
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='lower right')
        plt.show()

        return n_arms, accuracy, average_sample, majority_sample


class BanditAgent:
    def __init__(self):
        pass

    def init_display(self):
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.show()
        return ax1, ax2

class LILUCBAgent(BanditAgent):
    def __init__(self):
        BanditAgent.__init__(self)

    def run(self, sim, plot=False):
        beta = 1
        alpha = 9
        epsilon = 0.01
        delta = 0.0001

        if plot:
            ax1, ax2 = self.init_display()
        mean_array = np.zeros(sim.K)
        sample_count = np.ones(sim.K) * 3

        for i in range(0, sim.K):
            mean_array[i] = sim.sim(i) + sim.sim(i) + sim.sim(i)

        counter = 0
        finish = False
        while not finish:
            explore_val = np.zeros(sim.K)
            mean = np.zeros(sim.K)
            for i in range(0, sim.K):
                if sample_count[i] == 0:
                    explore_val[i] = +100000
                    mean = 0.0
                else:
                    n = sample_count[i]
                    temp = math.log(math.log((1 + epsilon) * n) / delta)
                    explore_val[i] = mean_array[i] / sample_count[i] + \
                                     (1 + beta) * (1 + math.sqrt(epsilon)) * math.sqrt(2 * (1 + epsilon) * temp / n)
                    mean[i] = mean_array[i] / sample_count[i]

            cur_sample = np.argmax(explore_val)
            mean_array[cur_sample] += sim.sim(cur_sample)
            sample_count[cur_sample] += 1

            sample_sum = np.sum(sample_count)

            if plot:
                counter += 1
                if counter % 20 == 0:
                    ax1.cla()
                    ax2.cla()
                    ax1.scatter(range(0, sim.K), explore_val, color='r')
                    ax1.scatter(range(0, sim.K), mean, color='b')
                    ax2.scatter(range(0, sim.K), sample_count, color='g')
                    plt.draw()
                    time.sleep(0.001)

            for i in range(0, sim.K):
                if sample_count[i] >= 1 + alpha * (sample_sum - sample_count[i]):
                    return i

    def ucb_risk(self, epsilon, delta):
        rou = (2.0 + epsilon) / epsilon * ((1 / math.log(1 + epsilon)) ** (1 + epsilon))
        return math.sqrt(rou * delta) + 4 * rou * delta / (1 - rou * delta)

    def ucb_delta(self, epsilon, risk):
        temp = risk * epsilon / 5 / (2 + epsilon)
        return temp ** (1 / (1 + epsilon))



class LILAEAgent(BanditAgent):
    def __init__(self):
        BanditAgent.__init__(self)
        self.a = 0.8
        self.b = 3.0
        self.c = 1.1

    def run(self, sim, plot=False):
        if plot:
            ax1, ax2 = self.init_display()

        index = np.zeros(sim.K)
        mean_array = np.zeros(sim.K)
        sample_count = np.zeros(sim.K)
        for i in range(0, sim.K):
            index[i] = i
            mean_array[i] += sim.sim(i) + sim.sim(i) + sim.sim(i) + sim.sim(i)
            sample_count[i] += 4

        counter = 0
        while True:
            mean = mean_array / sample_count
            boundary = np.sqrt(self.a * sample_count * np.log(np.log(sample_count) / np.log(self.c) + 1)
                               + self.b * sample_count) / sample_count
            n = sample_count[np.argmax(mean)]
            boundary[np.argmax(mean)] = np.sqrt(self.a * n * math.log(math.log(n, self.c) + 1) + self.b * n * math.log(mean_array.size, 1.3)) / n
            ucb = mean + boundary
            lcb = mean - boundary
            maxlcb = lcb[np.argmax(mean)]

            if np.min(ucb) <= maxlcb:
                for k in range(0, mean_array.size):
                    if ucb[k] <= maxlcb:
                        ucb = np.delete(ucb, k)
                        lcb = np.delete(lcb, k)
                        mean = np.delete(mean, k)
                        mean_array = np.delete(mean_array, k)
                        sample_count = np.delete(sample_count, k)
                        index = np.delete(index, k)
                        break
                if index.size == 1:
                    break

            cur_sample = np.argmax(ucb - lcb)
            mean_array[cur_sample] += sim.sim(index[cur_sample])
            sample_count[cur_sample] += 1

            if plot:
                counter += 1
                if counter % 500 == 0:
                    ax1.cla()
                    ax2.cla()
                    ax1.scatter(index, ucb, color='r')
                    ax1.scatter(index, mean, color='b')
                    ax1.scatter(index, lcb, color='c')
                    ax2.scatter(index, sample_count, color='g')
                    plt.yscale('log')
                    plt.draw()
                    time.sleep(0.001)

        return index[0]

if __name__ == '__main__':
    sim = Simulator(200, kind='H3')
    ae_agent = LILAEAgent()

    #ae_agent.run(sim, plot=True)
    #print(sim.nsamples)
    #sim.plot_samples()

    test = MABTestBench(kind='H2', size=10)
    test.test_and_plot(ae_agent)


