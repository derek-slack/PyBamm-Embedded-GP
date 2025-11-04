import numpy as np
import matplotlib.pyplot as plt

class PostProcess():
    """
    Class for processing results of embedded gp routine
    """
    def __init__(self, samples, BIC, data):
        """
            samples, BIC, are the csv results from the routine
            Data is a numpy vector of the comparision result
        """

        self.samples = np.loadtxt(samples)
        self.BIC = np.loadtxt(BIC)
        self.data = data

    def plot_betas(self):
        """
        creates plots for all beta terms against number of draws, see how each parameter moved through routine
        """
        for i in range(self.samples.shape[1]):
            beta_str = 'Beta' + str(i)
            plt.plot(self.samples[:,i])
            plt.title(beta_str)
            plt.xlabel('Draws')
            plt.show()

    def plot_NLL(self):
        """
        Plots Negative Log Likelihood against number of draws
        """
        plt.plot(self.BIC[1:])
        plt.xlabel('Draws')
        plt.ylabel('Negative Log Likelihood')
        plt.show()
    def average_samples(self, n):
        """
        Returns average of beta terms for use in evaluation
        """
        avg_betas = np.mean(self.samples[-n:,:], axis=0)
        return avg_betas