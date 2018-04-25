from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.basemap import Basemap

class Evaluater(object):
    """
    Class provides methods to evaluate the output data of the LSTM and to plot the results
    """

    def __init__(self):
        # limits for plotting space
        self.ymin = 150
        self.ymax = 350
        self.corr_all = []
        self.rmse_all = []


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
        shape = ypred.shape[1] * ypred.shape[2]
        s1 = ypred.shape[0]
        # statistics

        for t in range(ypred.shape[0]):
            yt = ytest[t,:,:]
            yp = ypred[t,:,:]
            self.rmse_all.append(np.sqrt(mean_squared_error(yt.reshape(shape), yp.reshape(shape))))
            self.corr_all.append(np.corrcoef(yt.reshape(shape), yp.reshape(shape))[0,1])

        rmse = np.mean(self.rmse_all)
        corr = np.mean(self.corr_all)

        shape = ypred.shape[0] * ypred.shape[1] * ypred.shape[2]

        # linear regression
        m, b = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(self.ymin-10, self.ymax+10, 1)
        yreg = np.add(np.multiply(m, x), b)

        # plotting
        print("plotting results as hist2d plot...")

        fig, ax = plt.subplots(figsize=(7, 4))

        # fig.suptitle(
        #     'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} ' \
        #     'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
        #         neurons,
        #         batch_size,
        #         epochs,
        #         time_steps,
        #         rmse, corr, runtime))

        plt.hist2d(ytest.reshape(shape), ypred.reshape(shape), bins=50, cmap='Greys', norm=LogNorm())
        plt.plot(x, yreg, '--', label="Regression", color='k', lw=1)

        cb = plt.colorbar()
        cb.set_label('number (1)')

        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xlabel("Test")
        ax.set_ylabel("Prediction")
        ax.set_xlim(self.ymin, self.ymax)
        ax.set_ylim(self.ymin, self.ymax)

        print("\t saving figure...")

        # plt.savefig("Images/LSTM_hist2d_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
        #             (neurons, batch_size, epochs, time_steps), dpi=400)

    def map_mae(self, ytest, ypred, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        """
        Plots the mean absolute error between the truth and prediction on the map.

        Args:
            ytest:
            ypred:
            neurons:
            batch_size:
            epochs:
            time_steps:
            runtime:
            path:

        Returns:

        """
        diff = ytest - ypred

        mae = np.mean(diff, axis=0)

        lat = np.linspace(-90, 90, 192)
        lon = np.linspace(-180, 180, 96)
        X, Y = np.meshgrid(lon, lat)

        ytest = np.reshape(ytest, (ytest.shape[0] * ytest.shape[1] * ytest.shape[2]))
        ypred = np.reshape(ypred, (ypred.shape[0] * ypred.shape[1] * ypred.shape[2]))

        rmse = np.sqrt(mean_squared_error(ytest, ypred))
        corr = np.corrcoef(ytest, ypred)

        # plotting
        print("plotting results as mae map plot...")

        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} ' \
            'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                rmse, corr[0, 1], runtime))

        earth = Basemap()
        earth.drawcoastlines()

        levels = np.linspace(-8.5, 8.5, 18, endpoint=True)
        ticks = np.linspace(-8, 8, 9, endpoint=True)

        cp = plt.contourf(X, Y, mae, cmap=cm.seismic, levels=levels, extend="both", alpha=0.9)
        cb = plt.colorbar(cp, ticks=ticks)
        cb.set_label(r'CMIP5 Temperature - Pred. Temperature ($\Delta{}$K)')

        #ax.set_xlabel("lat")
        #ax.set_ylabel("lon")

        plt.tight_layout()

        print("\t saving figure...")

        plt.savefig("Images/LSTM_maemap_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)