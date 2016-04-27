__author__ = 'shengjia'


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time
from scipy.special import zeta
from numpy.random import normal as nrand


class Simulator:
    """ This class simulates bandit with requested arm count, distribution, and mean. Also pulls to each arm is recorded
    and can be plotted or returned if needed """

    def __init__(self, nbr_arms, kind='H2', dist='bernoulli'):
        self.K = nbr_arms
        self.nsamples = 0
        self.kind = kind
        self.sample_size = np.zeros(self.K)
        self.dist = dist
        self.mean_list = np.zeros(self.K)

        # Compute the mean for each arm
        for arm in range(0, self.K):
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
        """ Draw a sample from given arm """
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
        """ Return the H1 hardness of current configuration """
        hardness_sum = 0.0
        for arm in range(1, self.K):
            hardness_sum += 1.0 / ((self.mean_list[0] - self.mean_list[arm]) ** 2)
        return hardness_sum

    def finish(self):
        """ Reset internal states and return number of samples requested so far """
        copy = self.nsamples
        self.sample_size = np.zeros(self.K)
        self.nsamples = 0
        return copy

    def plot_samples(self):
        """ Plot the samples requested for each arm """
        plt.clf()
        plt.scatter(range(0, self.K), self.sample_size)
        plt.yscale('log')
        plt.ioff()
        plt.show()


class MABTestBench:
    def __init__(self, kind, size):
        self.kind = kind
        self.size = size
        self.k_range = 10

    def test(self, agent):
        """ Run the test on an agent and return the number of arms, average accuracy, average number of samples, and number
    of samples top 10% of runs collected """
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
            K = int(K * 1.5)
        return narms_array, accuracy, average_sample, majority_sample

    def test_and_plot(self, agent):
        """ Run test on an agent and plot its performance """
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
        """ Compare a list of agents and plot their sample complexity """
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        type_list = ['-', '--', '-.', ':']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

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
    """ Base class for all mab agents """
    
    def __init__(self):
        self.ax1 = None
        self.ax2 = None

    def run(self, sim):
        """ All agents must inherit this method """
        return 0

    def init_display(self):
        """ Call this if we want to graphical visualize the agent's behavior """
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.show()
        self.ax1 = ax1
        self.ax2 = ax2
        return ax1, ax2

    def draw(self, x_index, left_objects, right_objects):
        """ Draw non-blocking on the left axis left_objects and right axis right_objects,
        use x_index as x coordinates """
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
        self.ax1.set_ylim((0, 1))
        self.ax2.set_ylim((0, 1))
        plt.draw()
        time.sleep(0.001)


class LILAgent(BanditAgent):
    """ Base class for all agents based on LIL bounds """
    
    def __init__(self, confidence):
        BanditAgent.__init__(self)
        self.confidence = confidence
        delta = 1.0
        while self.risk(delta) > self.confidence:
            delta /= 1.2
        self.delta = delta

    def risk(self, delta, epsilon = 0.01):
        """ return the risk the bound do not hold given delta """
        rou = 2 * (2.0 + epsilon) / epsilon * ((1 / math.log(1 + epsilon)) ** (1 + epsilon))
        return rou * delta

    def boundary(self, n, delta, epsilon=0.01):
        """ Compute the LIL bound given n, delta """
        temp = np.log((np.log(1 + epsilon) + np.log(n)) / delta)
        return np.sqrt((1 + epsilon) * temp / 2 / n) * (1 + np.sqrt(epsilon))


class LILUCBAgent(LILAgent):
    """ Agent that uses LIL-UCB algorithm """

    def __init__(self, confidence):
        LILAgent.__init__(self, confidence)

    def run(self, sim, plot=False):
        beta = 1.0
        alpha = 9.0
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
    """ Agent that uses LIL-LUCB algorithm """

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


class AHRAgent(BanditAgent):
    """ Agent that uses LIL-AE algorithm """

    def __init__(self, confidence, stop_interval=1):
        BanditAgent.__init__(self)
        self.confidence = confidence / 2
        self.stop_interval = stop_interval
        if stop_interval != 1:
            self.a = 0.55
            self.c = self.stop_interval
            self.b = (math.log(zeta(2 * self.a, 1)) - math.log(confidence)) / 2
        else:
            self.a = 0.6
            self.c = 1.1
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
        next_stop = 1
        while True:
            counter += 1
            mean = mean_array / sample_count
            if self.stop_interval == 1:
                conf_c = self.c
            else:
                conf_c = 1.0
            boundary = np.sqrt(self.a * sample_count * np.log(np.log(sample_count) / np.log(self.c) + 1)
                               + self.b * sample_count) / sample_count
            n = sample_count[np.argmax(mean)]
            boundary[np.argmax(mean)] = np.sqrt(self.a * n * math.log(math.log(n, self.c) + 1) +
                                                (self.b + conf_c * math.log(mean_array.size) / 2) * n) / n
            ucb = mean + boundary
            lcb = mean - boundary
            maxlcb = lcb[np.argmax(mean)]

            if self.stop_interval != 1:
                if counter < next_stop:
                    continue
                while next_stop < n + 1:
                    next_stop *= self.stop_interval
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

            if plot and counter % 20 == 0:
                self.draw(index, [ucb, mean, lcb], [sample_count])

        return index[0]


class LILLSAgent(LILAgent):
    """ Agent that uses LIL-UCB + LS """

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


class MedianAgent(BanditAgent):
    def __init__(self, confidence):
        BanditAgent.__init__(self)
        self.confidence = confidence

    def run(self, sim):
        arms = np.zeros(sim.K)
        for i in range(0, sim.K):
            arms[i] = i
        r = 0
        h = 1
        self.init_display()
        while True:
            r += 1
            if arms.size == 1:
                return arms[0]
            epsilon_r = 2 ** (-r)
            delta_r = self.confidence / 50 / (r ** 2)
            print("Step 1")
            a_r = self.median_elimination(sim, arms, epsilon_r / 4, 0.01)
            print("Step 2")
            mu_a_r = self.uniform_sampling(sim, np.asarray([a_r]), epsilon_r / 4, delta_r)[0]
            print("Step 3")
            if self.frac_test(sim, arms, mu_a_r - 1.5 * epsilon_r, mu_a_r - 1.25 * epsilon_r, delta_r, 0.4, 0.1):
                print("Step 4")
                delta_h = self.confidence / 50 / (h ** 2)
                b_r = self.median_elimination(sim, arms, epsilon_r / 4, delta_h)
                print("Step 5")
                mu_b_r = self.uniform_sampling(sim, np.asarray([b_r]), epsilon_r / 4, delta_h)[0]
                print("Step 6")
                arms = self.elimination(sim, arms, mu_b_r - 0.5 * epsilon_r, mu_b_r - 0.25 * epsilon_r, delta_h)
                h += 1
            print("One iteration completed, with " + str(arms.size) + " arms left")
            self.draw(arms, np.zeros(arms.size), np.ones(arms.size))
            time.sleep(2)

    def uniform_sampling(self, sim, arms, epsilon, delta):
        repeat = int(2 / (epsilon ** 2) * math.log(2 / delta))
        mean_array = np.zeros(arms.size)
        for index in range(0, arms.size):
            for i in range(0, repeat):
                mean_array[index] += sim.sim(arms[index])
            mean_array[index] /= repeat
        return mean_array

    def frac_test(self, sim, arms, c_l, c_r, delta, t, epsilon):
        count = 0.0
        total = int(math.log(2 / delta) / ((epsilon / 3) ** 2) / 2)
        for i in range(0, total):
            arm = random.choice(arms)
            mu = self.uniform_sampling(sim, np.asarray([arm]), (c_r - c_l) / 2.0, epsilon / 3.0)[0]
            if mu < (c_l + c_r) / 2:
                count += 1.0
        if count / float(total) > t:
            return True
        else:
            return False

    def elimination(self, sim, arms, c_l, c_r, delta):
        c_m = (c_l + c_r) / 2.0
        r = 0
        while True:
            r += 1
            delta_r = delta / 10.0 / (2.0 ** r)
            print("Ministep " + str(r))
            if self.frac_test(sim, arms, c_l, c_m, delta_r, 0.075, 0.025):
                mean_array = self.uniform_sampling(arms, (c_r-c_m) / 2, delta_r)
                thresh = (c_m + c_r) / 2
                next_arms = []
                for arm, mean in zip(arms, mean_array):
                    if mean > thresh:
                        next_arms.append(arm)
                arms = np.asarray(next_arms)
            else:
                return arms

    def median_elimination(self, sim, arms, epsilon, delta):
        epsilon /= 4.0
        delta /= 2.0
        while True:
            mean_array = np.zeros(arms.size)
            repeat = int(1.0 / ((epsilon / 2.0) ** 2) * math.log(3.0/delta))
            for index, arm in zip(range(0, arms.size), arms):
                for rep in range(0, repeat):
                    mean_array[index] += sim.sim(arm)
                mean_array[index] /= repeat
                # self.draw(arms, [mean_array], [mean_array])
            sorted_array = mean_array.copy()
            sorted_array.sort()
            median = sorted_array[mean_array.size / 2]    # This selects median rounded up

            # self.draw(arms, [mean_array], [np.ones(arms.size) * median])
            next_arms = []
            for index in range(0, arms.size):
                if mean_array[index] >= median:
                    next_arms.append(arms[index])
            arms = np.asarray(next_arms)

            if arms.size == 1:
                break
            epsilon = epsilon * 3.0 / 4.0
            delta /= 2.0
        return arms[0]


if __name__ == '__main__':
    ae_agent = AHRAgent(0.05)
    ucb_agent = LILUCBAgent(0.05)
    ls_agent = LILLSAgent(0.05)
    lucb_agent = LILLUCBAgent(0.05)
    median_agent = MedianAgent(0.05)
    eae_agent = AHRAgent(0.05, stop_interval=1.05)
    #test_sim = Simulator(20)
    #eae_agent.run(test_sim, plot=True)
    #median_agent.run(test_sim)
    #median_agent.median_elimination(test_sim, np.asarray(range(0, 20)), 0.1, 0.05)
    test = MABTestBench(kind='H2', size=1)

    agents = [ae_agent, eae_agent, ucb_agent, ls_agent, lucb_agent]
    labels = ['AHR', 'ES-AHR', 'LIL-UCB', 'LIL-UCB + LS', 'LIL-LUCB']
    test.compare_and_plot(agents, labels)



