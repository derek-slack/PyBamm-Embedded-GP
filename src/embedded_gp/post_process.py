import numpy as np
import matplotlib.pyplot as plt

class PostProcess():
    def __init__(self, samples, BIC, data):
        self.samples = np.loadtxt(samples)
        self.BIC = np.loadtxt(BIC)
        self.data = data

    def plot_betas(self):
        for i in range(self.samples.shape[1]):
            beta_str = 'Beta' + str(i)
            plt.plot(self.samples[:,i])
            plt.title(beta_str)
            plt.xlabel('Draws')
            plt.show()

    def plot_NLL(self):
        plt.plot(self.BIC[1:])
        plt.xlabel('Draws')
        plt.ylabel('Negative Log Likelihood')
        plt.show()
    def average_samples(self, n):
        avg_betas = np.mean(self.samples[-n:,:], axis=0)
        return avg_betas