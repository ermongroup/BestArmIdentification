__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
from scipy.special import zeta
from numpy.random import normal as nrand

class Simulator:
    def __init__(self, K, kind='H2', dist='bernoulli'):
        self.K = K
        self.nsamples = 0
        self.kind = kind
        self.sample_size = np.zeros(K)
        self.dist = dist
        self.mean_list = np.zeros(K)
        for arm in range(0, K):
            if self.kind == 'H0':
                if arm == 0:
                    self.mean_list[arm] = 0.6
                else:
                    self.mean_list[arm] = 0.3
            elif self.kind == 'H2':
                self.mean_list[arm] = 1.0 - (float(arm) / self.K) ** 0.6
                if self.mean_list[arm] < 0:
                    self.mean_list[arm] = 0
            elif self.kind == 'H3':
                self.mean_list[arm] = 1.0 - float(arm) / self.K
                if self.mean_list[arm] < 0:
                    self.mean_list[arm] = 0.0
            elif self.kind == 'H':
                self.mean_list[arm] = 0.5

    def sim(self, arm):
        self.nsamples += 1
        self.sample_size[arm] += 1
        if self.dist == 'bernoulli':
            if random.random() > self.mean_list[arm]:
                return 0
            else:
                return 1
        else:
            return nrand(self.mean_list[arm], 0.25)

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
        self.k_range = 7

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
                print("    " + str(rep) + "-th iteration"),
                sim = Simulator(K, self.kind, dist='gaussian')
                if agent.run(sim) == 0:
                    correct_count += 1
                    print("correct"),
                else:
                    print("incorrect"),
                sample_count[rep] = sim.finish()
                print(str(sample_count[rep] / sim.hardness()) + " samples")

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

    def compare_and_plot(self, agents, labels):
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        type_list = ['-', '--', '-.', ':']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #ax2 = ax1.twinx()
        #plt.hold(True)
        #plt.title('Comparison of Required Sample Size')

        counter = 0
        lns = []
        for agent in agents:
            mean, accuracy, average_sample, majority_sample = self.test(agent)
            lns += ax1.plot(mean, average_sample,
                            type_list[counter % len(type_list)],
                            c=color_list[counter % len(color_list)],
                            label=labels[counter], linewidth=4)
            counter += 1

        ax1.set_xlim((min(mean), max(mean)))
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        ax1.set_xlabel('nbr of arms', fontsize=16)
        ax1.set_ylabel('nbr of samples (in units of H1)', fontsize=16)

        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper right', fontsize=16)
        plt.show()

class BanditAgent:
    def __init__(self):
        self.ax1 = None
        self.ax2 = None

    def init_display(self):
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.show()
        self.ax1 = ax1
        self.ax2 = ax2
        return ax1, ax2

    def draw(self, x_index, left_objects, right_objects):
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        color_count = 0
        self.ax1.cla()
        self.ax2.cla()
        for item in left_objects:
            self.ax1.scatter(x_index, item, color=color_list[color_count % len(color_list)])
            color_count += 1
        for item in right_objects:
            self.ax2.scatter(x_index, item, color=color_list[color_count % len(color_list)])
            color_count += 1
        plt.draw()
        time.sleep(0.001)

class LILAgent(BanditAgent):
    def __init__(self, confidence):
        BanditAgent.__init__(self)
        self.confidence = confidence
        delta = 1.0
        while self.risk(delta) > self.confidence:
            delta /= 1.2
        self.delta = delta

    def risk(self, delta, epsilon = 0.01):
        rou = 2 * (2.0 + epsilon) / epsilon * ((1 / math.log(1 + epsilon)) ** (1 + epsilon))
        return rou * delta

    def boundary(self, n, delta, epsilon=0.01):
        temp = np.log((np.log(1 + epsilon) + np.log(n)) / delta)
        return np.sqrt((1 + epsilon) * temp / 2 / n) * (1 + np.sqrt(epsilon))


class LILUCBAgent(LILAgent):
    def __init__(self, confidence):
        LILAgent.__init__(self, confidence)

    def run(self, sim, plot=False):
        beta = 1
        alpha = 9
        delta = self.delta

        if plot:
            self.init_display()
        mean_array = np.zeros(sim.K)
        sample_count = np.ones(sim.K) * 3
        for i in range(0, sim.K):
            mean_array[i] = sim.sim(i) + sim.sim(i) + sim.sim(i)

        counter = 0
        finish = False
        while not finish:
            mean = mean_array / sample_count
            explore_val = mean + (1 + beta) * self.boundary(sample_count, delta)

            cur_sample = np.argmax(explore_val)
            mean_array[cur_sample] += sim.sim(cur_sample)
            sample_count[cur_sample] += 1

            counter += 1
            if plot and counter % 20 == 0:
                self.draw(range(0, sim.K), [explore_val, mean], [sample_count])

            sample_sum = np.sum(sample_count)
            for i in range(0, sim.K):
                if sample_count[i] >= 1 + alpha * (sample_sum - sample_count[i]):
                    return i


class LILLUCBAgent(LILAgent):
    def __init__(self, confidence):
        LILAgent.__init__(self, confidence)

    def run(self, sim, plot=False):
        delta = self.delta

        if plot:
            self.init_display()

        mean_array = np.zeros(sim.K)
        sample_count = np.ones(sim.K) * 3
        for i in range(0, sim.K):
            mean_array[i] = sim.sim(i) + sim.sim(i) + sim.sim(i)

        counter = 0
        finish = False
        while not finish:
            mean = mean_array / sample_count
            ucb = mean + self.boundary(sample_count, delta/sim.K)

            best = np.argmax(mean)
            mean_array[best] += sim.sim(best)
            sample_count[best] += 1

            temp = ucb[best]
            ucb[best] = -1000000
            cur_sample = np.argmax(ucb)
            mean_array[cur_sample] += sim.sim(cur_sample)
            sample_count[cur_sample] += 1
            ucb[best] = temp

            # Inspect LIL stopping criteria
            stop = True
            lcb = mean_array[best] / sample_count[best] - self.boundary(sample_count[best], delta/sim.K)
            for j in range(0, sim.K):
                if j == best:
                    continue
                if lcb < ucb[j]:
                    stop = False
                    break
            if stop:
                return best

            counter += 1
            if plot and counter % 50 == 0:
                self.draw(range(0, sim.K), [ucb, mean], [sample_count])


class LILAEAgent(BanditAgent):
    def __init__(self, confidence):
        BanditAgent.__init__(self)
        self.a = 0.6
        self.c = 1.1
        self.confidence = confidence / 2
        self.b = (math.log(zeta(2 * self.a / self.c, 1)) - math.log(confidence)) * self.c / 2

    def run(self, sim, plot=False):
        if plot:
            self.init_display()

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
            boundary[np.argmax(mean)] = np.sqrt(self.a * n * math.log(math.log(n, self.c) + 1) +
                                                (self.b + self.c * math.log(mean_array.size) / 2) * n) / n
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

            counter += 1
            if plot and counter % 20 == 0:
                self.draw(index, [ucb, mean, lcb], [sample_count])

        return index[0]

class LILLSAgent(LILAgent):
    def __init__(self, confidence):
        LILAgent.__init__(self, confidence / 2)

    def run(self, sim, plot=False):
        beta = 1
        alpha = 9
        delta = self.delta

        if plot:
            self.init_display()

        mean_array = np.zeros(sim.K)
        sample_count = np.ones(sim.K) * 3
        for i in range(0, sim.K):
            mean_array[i] = sim.sim(i) + sim.sim(i) + sim.sim(i)

        counter = 0
        finish = False
        while not finish:
            mean = mean_array / sample_count
            explore_val = mean + (1 + beta) * self.boundary(sample_count, delta)
            ucb = mean + self.boundary(sample_count, delta/sim.K)

            cur_sample = np.argmax(explore_val)
            mean_array[cur_sample] += sim.sim(cur_sample)
            sample_count[cur_sample] += 1

            sample_sum = np.sum(sample_count)

            for i in range(0, sim.K):
                if sample_count[i] >= 1 + alpha * (sample_sum - sample_count[i]):
                    return i

            # Inspect LIL stopping criteria
            stop = True
            best = np.argmax(mean)
            lcb = mean_array[best] / sample_count[best] - self.boundary(sample_count[best], delta/sim.K)
            for j in range(0, sim.K):
                if j == best:
                    continue
                if lcb < ucb[j]:
                    stop = False
                    break
            if stop:
                return best

            counter += 1
            if plot and counter % 20 == 0:
                self.draw(range(0, sim.K), [explore_val, ucb, mean], [sample_count])


if __name__ == '__main__':
    sim = Simulator(20, kind='H0')
    ae_agent = LILAEAgent(0.05)

    ucb_agent = LILUCBAgent(0.05)

    ls_agent = LILLSAgent(0.05)

    lucb_agent = LILLUCBAgent(0.05)

    test = MABTestBench(kind='H2', size=20)

    #test.test_and_plot(ae_agent)

    agents = [ae_agent, ucb_agent, ls_agent, lucb_agent]
    labels = ['LIL-AE', 'LIL-UCB', 'LIL-UCB + LS', 'LIL-LUCB']
    test.compare_and_plot(agents, labels)



