from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from datetime import datetime
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm, PowerNorm


class Evaluater(object):
    """
    Class provides methods to evaluate the output data of the LSTM and to plot the results
    """

    def __init__(self):
        # limits for plotting space
        self.ymin = 150
        self.ymax = 350

    def scatter(self, ytest, ypred, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        """
        Creates a scatterplot plotting the truth vs. the prediction.
        Furthermore plots a regression line and writes some stats in the tilte.

        Args:
            ytest: numpy array, truth
            ypred: numpy array, prediction
            runtime: int, time of model run in seconds
        """

        shape = ypred.shape[0] * ypred.shape[1]

        rmse = np.sqrt(mean_squared_error(ytest.reshape(shape), ypred.reshape(shape)))
        corr = np.corrcoef(ytest.reshape(shape), ypred.reshape(shape))

        m, b = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(self.ymin, self.ymax, 1)
        yreg = np.add(np.multiply(m, x), b)

        print("plotting results as scatter plot...")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} '\
            'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                rmse, corr[0, 1],runtime))
        ax.plot(ytest.reshape(shape), ypred.reshape(shape), lw=0, marker=".", color="blue", alpha=0.05,
                markeredgewidth=0.0)
        ax.plot(x, yreg, '-', label="Regression", color="red", lw=2)
        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xlabel("Test")
        ax.set_ylabel("Prediction")
        ax.set_xlim(self.ymin, self.ymax)
        ax.set_ylim(self.ymin, self.ymax)
        print("\t saving figure...")
        plt.savefig("Images/LSTM_scatter_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)

    def hist2d(self, ytest, ypred, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        """
        Creates a 2d histogram plotting the truth vs. the prediction.
        Furthermore plots a regression line and writes some stats in the tilte.

        Args:
            ytest: numpy array, truth
            ypred: numpy array, prediction
            neurons: int, number of neurons in LSTM
            batch_size: int, batch size in LSTM
            epochs: int, number of epochs in LSTM
            time_steps: int, number of time steps in LSTM
            runtime: float, runtime of model run in seconds
            path: str, path for output
        """
        shape = ypred.shape[0] * ypred.shape[1]

        # statistics
        rmse = np.sqrt(mean_squared_error(ytest.reshape(shape), ypred.reshape(shape)))
        corr = np.corrcoef(ytest.reshape(shape), ypred.reshape(shape))

        # linear regression
        m, b = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(self.ymin-10, self.ymax+10, 1)
        yreg = np.add(np.multiply(m, x), b)

        # plotting
        print("plotting results as hist2d plot...")

        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} ' \
            'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                rmse, corr[0, 1], runtime))

        plt.hist2d(ytest.reshape(shape), ypred.reshape(shape), bins=50, cmap='Greys', norm=LogNorm())
        plt.plot(x, yreg, '--', label="Regression", color='k', lw=1)

        plt.colorbar()

        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xlabel("Test")
        ax.set_ylabel("Prediction")
        ax.set_xlim(self.ymin, self.ymax)
        ax.set_ylim(self.ymin, self.ymax)

        print("\t saving figure...")

        plt.savefig("Images/LSTM_hist2d_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)
