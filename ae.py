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

    def sim(self, arm):
        self.sample_size[arm] += 1
        self.nsamples += 1
        if self.kind == 'H0':
            if arm == 0:
                mean = 0.6
            else:
                mean = 0.3
        elif self.kind == 'H2':
            mean = 0.9 - (float(arm) / self.K) ** 0.6
        elif self.kind == 'H3':
            if arm == 0:
                mean = 0.5
            else:
                mean = 0.5
        if random.random() > mean:
            return 0
        else:
            return 1

    def plot_samples(self):
        plt.clf()
        plt.scatter(range(0, self.K), self.sample_size)
        plt.yscale('log')
        plt.ioff()
        plt.show()

class LILAEAgent:
    def __init__(self):
        self.a = 0.8
        self.b = 3.0
        self.c = 1.1

    def init_display(self):
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.show()
        return ax1, ax2

    def run_ae(self, sim):
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

    def ucb_risk(self, epsilon, delta):
        rou = (2.0 + epsilon) / epsilon * ((1 / math.log(1 + epsilon)) ** (1 + epsilon))
        return math.sqrt(rou * delta) + 4 * rou * delta / (1 - rou * delta)

    def ucb_delta(self, epsilon, risk):
        temp = risk * epsilon / 5 / (2 + epsilon)
        return temp ** (1 / (1 + epsilon))

    def run_ucb(self, sim):
        beta = 1
        alpha = 9
        epsilon = 0.01
        delta = 0.0001

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
                    finish = True



if __name__ == '__main__':
    sim = Simulator(200, kind='H3')
    agent = Agent()

    agent.run_ae(sim)
    print(sim.nsamples)
    sim.plot_samples()

    delta_array = []
    risk_array = []
    for delta_index in range(0, 100):
        delta = delta_index * 0.0001
        delta_array.append(delta)
        risk_array.append(agent.ucb_risk(0.01, delta))
    plt.plot(delta_array, risk_array)
    plt.show()

    risk_array = []
    delta_array = []
    for risk_index in range(0, 100):
        risk = risk_index * 0.01
        delta_array.append(agent.ucb_delta(0.01, risk))
        risk_array.append(risk)
    plt.plot(risk_array, delta_array)
    plt.show()

