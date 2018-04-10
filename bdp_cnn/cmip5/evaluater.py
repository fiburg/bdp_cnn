from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error


class Evaluater(object):
    """
    Class provides methods to evaluate the output data of the LSTM and to plot the results
    """

    def analysis_scatter(self, ytest, ypred, runtime):
        """
        Creates a scatterplot plotting the truth vs. the prediction.
        Furthermore plots a regression line and writes some stats in the tilte.

        Args:
            truth: numpy array
            preds: numpy array

        Returns:

        """

        shape = ypred.shape[0] * ypred.shape[1]

        rmse = np.sqrt(mean_squared_error(ytest.reshape(shape), ypred.reshape(shape)))
        corr = np.corrcoef(ytest.reshape(shape), ypred.reshape(shape))

        m, b = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(100, 400, 1)
        yreg = np.add(np.multiply(m, x), b)

        print("plotting Results...")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} '\
            'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
                self.neurons,
                self.batch_size,
                self.nb_epoch,
                self.time_steps,
                rmse, corr[0, 1],runtime))
        ax.plot(ytest.reshape(shape), ypred.reshape(shape), lw=0, marker=".", color="blue", alpha=0.05,
                markeredgewidth=0.0)
        ax.plot(x, yreg, '-', label="Regression", color="red", lw=2)
        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xlabel("Test")
        ax.set_ylabel("Prediction")
        ax.set_xlim(150, 350)
        ax.set_ylim(150, 350)
        print("\t saving figure...")
        plt.savefig("Images/LSTM_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (self.neurons, self.batch_size, self.nb_epoch, self.time_steps), dpi=400)
