from netCDF4 import Dataset
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from bdp_cnn.cmip5.datahandler import DataHandler
import glob

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
        self.bias_all = []
        self.std_all = []

    def calc_corr(self, ytest, ypred):
        shape = ypred.shape[1] * ypred.shape[2]

        for t in range(ypred.shape[0]):
            yt = ytest[t,:,:]
            yp = ypred[t,:,:]

            self.corr_all.append(np.corrcoef(yt.reshape(shape), yp.reshape(shape))[0,1])

        return np.mean(self.corr_all)

    def calc_rmse(self, ytest, ypred):

        for t in range(ypred.shape[0]):
            diff = ypred[t, :, :] - ytest[t, :, :]

            rmse = np.sqrt(np.mean(np.square(diff), axis=(0, 1)))

            self.rmse_all.append(rmse)

        return np.mean(self.rmse_all)

    def calc_bias(self, ytest, ypred):
        diff = ypred - ytest

        bias = np.mean(diff, axis=(1,2))

        for t in range(ypred.shape[0]):
            self.bias_all.append(bias[t])

        return np.mean(self.bias_all)

    def calc_std(self, ytest, ypred):
        ypred_mean = np.mean(ypred, axis=(1, 2))
        ytest_mean = np.mean(ytest, axis=(1, 2))

        for t in range(ypred.shape[0]):
            ypred[t, :, :] = ypred[t, :, :] - ypred_mean[t]
            ytest[t, :, :] = ytest[t, :, :] - ytest_mean[t]

            diff_unbiased = ypred[t, :, :] - ytest[t, :, :]

            std = np.sqrt(np.mean(np.square(diff_unbiased), axis=(0,1)))

            self.std_all.append(std)

        return np.mean(self.std_all)


    def scatter(self, ytest, ypred, neurons, batch_size, epochs, time_steps, runtime, path=''):
        """
        Creates a scatterplot plotting the truth vs. the prediction.
        Furthermore plots a regression line and writes some stats in the tilte.

        Args:
            ytest: numpy array, truth
            ypred: numpy array, prediction
            runtime: int, time of model run in seconds
        """
        rmse = self.calc_rmse(ytest, ypred)
        corr = self.calc_corr(ytest, ypred)

        shape = ytest.shape[0], ytest.shape[1] * ytest.shape[2]

        reg = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(self.ymin, self.ymax, 1)
        yreg = np.add(np.multiply(reg[0], x), reg[1])

        print("plotting results as scatter plot...")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} '\
            'and CORR = {5:.3f}, runtime = {6:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                rmse, corr, runtime))
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
        plt.savefig(path+"LSTM_scatter_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
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
        rmse = self.calc_rmse(ytest, ypred)
        corr = self.calc_corr(ytest, ypred)

        shape = ypred.shape[0] * ypred.shape[1] * ypred.shape[2]

        # linear regression
        reg = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(self.ymin-10, self.ymax+10, 1)
        yreg = np.add(np.multiply(reg[0], x), reg[1])

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
                rmse, corr, runtime))

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

        plt.savefig(path+"LSTM_hist2d_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)

    def map_mae(self, ytest, ypred, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        """
        Plots the mean absolute error between the truth and prediction on the map.

        Args:
            ytest: numpy array: truth
            ypred: numpy array: prediction
            neurons: int: number of neurons in LSTM
            batch_size: int: batch size in LSTM
            epochs: int: number of epochs in LSTM
            time_steps: int: number of time steps in LSTM
            runtime: float: runtime of model run in seconds
            path: str: path for output
        """
        diff = ypred - ytest

        mae = np.mean(diff, axis=0)

        lat = [88.57216851,  86.72253095,  84.86197029,  82.99894164,  81.13497684,
                79.27055903,  77.40588808,  75.54106145,  73.67613231,  71.81113211,
                69.94608065,  68.08099099,  66.21587211,  64.35073041,  62.48557052,
                60.62039593,  58.75520927,  56.8900126,   55.02480754,  53.15959537,
                51.29437714,  49.4291537,   47.56392575,  45.69869388,  43.83345858,
                41.96822027,  40.1029793,   38.23773599,  36.37249059,  34.50724334,
                32.64199444,  30.77674406,  28.91149237,  27.0462395,   25.18098558,
                23.31573073,  21.45047504,  19.58521861,  17.71996153,  15.85470387,
                13.98944571,  12.12418712,  10.25892817,   8.39366891,   6.5284094,
                4.66314971,   2.79788988,   0.93262997, - 0.93262997, - 2.79788988,
                - 4.66314971, - 6.5284094, - 8.39366891, - 10.25892817, - 12.12418712,
                - 13.98944571, - 15.85470387, - 17.71996153, - 19.58521861, - 21.45047504,
                - 23.31573073, - 25.18098558, - 27.0462395, - 28.91149237, - 30.77674406,
                - 32.64199444, - 34.50724334, - 36.37249059, - 38.23773599, - 40.1029793,
                - 41.96822027, - 43.83345858, - 45.69869388, - 47.56392575, - 49.4291537,
                - 51.29437714, - 53.15959537, - 55.02480754, - 56.8900126, - 58.75520927,
                - 60.62039593, - 62.48557052, - 64.35073041, - 66.21587211, - 68.08099099,
                - 69.94608065, - 71.81113211, - 73.67613231, - 75.54106145, - 77.40588808,
                - 79.27055903, - 81.13497684, - 82.99894164, - 84.86197029, - 86.72253095,
                - 88.57216851]

        lon = [0.,       1.875,    3.75,     5.625,    7.5,      9.375,   11.25,    13.125,
                15.,      16.875,   18.75,    20.625,   22.5,     24.375,   26.25,    28.125,
                30.,      31.875,   33.75,    35.625,   37.5,     39.375,   41.25,    43.125,
                45.,      46.875,   48.75,    50.625,   52.5,     54.375,   56.25,    58.125,
                60.,      61.875,   63.75,    65.625,   67.5,     69.375,   71.25,    73.125,
                75.,      76.875,   78.75,    80.625,   82.5,     84.375,   86.25,    88.125,
                90.,      91.875,   93.75,    95.625,   97.5,     99.375,  101.25,   103.125,
                105.,     106.875,  108.75,   110.625,  112.5,    114.375,  116.25,   118.125,
                120.,     121.875,  123.75,   125.625,  127.5,    129.375,  131.25,   133.125,
                135.,     136.875,  138.75,   140.625,  142.5,    144.375,  146.25,   148.125,
                150.,     151.875,  153.75,   155.625,  157.5,    159.375,  161.25,   163.125,
                165.,     166.875,  168.75,   170.625,  172.5,    174.375,  176.25,   178.125,
                180.,     181.875,  183.75,   185.625,  187.5,    189.375,  191.25,   193.125,
                195.,     196.875,  198.75,   200.625,  202.5,    204.375,  206.25,   208.125,
                210.,     211.875,  213.75,   215.625,  217.5,    219.375,  221.25,   223.125,
                225.,     226.875,  228.75,   230.625,  232.5,    234.375,  236.25,   238.125,
                240.,     241.875,  243.75,   245.625,  247.5,    249.375,  251.25,   253.125,
                255.,     256.875,  258.75,   260.625,  262.5,    264.375,  266.25,   268.125,
                270.,     271.875,  273.75,   275.625,  277.5,    279.375,  281.25,   283.125,
                285.,     286.875,  288.75,   290.625,  292.5,    294.375,  296.25,   298.125,
                300.,     301.875,  303.75,   305.625,  307.5,    309.375,  311.25,   313.125,
                315.,     316.875,  318.75,   320.625,  322.5,    324.375,  326.25,   328.125,
                330.,     331.875,  333.75,   335.625,  337.5,    339.375,  341.25,   343.125,
                345.,     346.875,  348.75,   350.625,  352.5,    354.375,  356.25,   358.125]

        ytest = np.reshape(ytest, (ytest.shape[0] * ytest.shape[1] * ytest.shape[2]))
        ypred = np.reshape(ypred, (ypred.shape[0] * ypred.shape[1] * ypred.shape[2]))

        rmse = np.sqrt(mean_squared_error(ytest, ypred))
        corr = np.corrcoef(ytest, ypred)

        # plotting
        print("plotting results as mae map plot...")

        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4} ' \
            'and CORR = {5}, runtime = {6} s'.format(
                str(neurons),
                str(batch_size),
                str(epochs),
                str(time_steps),
                rmse, corr[0, 1], str(runtime)))

        earth = Basemap(llcrnrlon=0., urcrnrlon=358.125, llcrnrlat=-88.57216851, urcrnrlat=88.57216851)
        earth.drawcoastlines()

        levels = np.linspace(-2.25, 2.25, 10, endpoint=True)

        ticks = np.linspace(-2, 2, 5, endpoint=True)

        x, y = earth(lon, lat)

        #cp = plt.contourf(x, y, mae.T, cmap=cm.seismic, levels=levels, extend="both")
        #print(mae.shape)


        cp = plt.imshow(mae.T, cmap=plt.cm.get_cmap("seismic", 9), vmin=-2.25, vmax=2.25, extent=[0.,358.125,-88.57216851,88.57216851],
                   interpolation='none')
        cb = plt.colorbar(cp, extend="both", ticks=ticks)
        cb.set_label(r'LSTM Temperature - CMIP5 Temperature ($\Delta{}$K)')

        plt.tight_layout()

        print("\t saving figure...")

        plt.savefig(path+"LSTM_maemap_"
                    +str(neurons)+"neurons_"
                    +str(batch_size)+"batchsize_"
                    +str(epochs)+"epochs_"
                    +str(time_steps)+"timesteps.png", dpi=400)

    def model_loss(self, loss, val_loss, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n ' \
            'and runtime = {4:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                runtime))

        xep = np.arange(epochs)+1

        ax.plot(xep, loss, label='loss')
        ax.plot(xep, val_loss, label='val_loss')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.legend()
        ax.grid()

        plt.savefig(path + "LSTM_losshistory_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)

    def model_lr(self, lr, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n ' \
            'and runtime = {4:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                runtime))

        xep = np.arange(epochs)+1

        ax.plot(xep, lr)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('epochs')
        ax.set_ylabel('learning rate')
        ax.grid()

        plt.savefig(path + "LSTM_lrhistory_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)

    def global_mean(self, time, data, neurons, batch_size, epochs, time_steps, runtime, path="./"):
        pass
        """
        ytest = data[1]
        ypred = data[2]

        test_mean = np.mean(ytest, axis=1)
        test_mean = np.mean(test_mean, axis=1)

        pred_mean = np.mean(ytest, axis=1)
        pred_mean = np.mean(pred_mean, axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))

        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n ' \
            'and runtime = {4:.2f} s'.format(
                neurons,
                batch_size,
                epochs,
                time_steps,
                runtime))

        xep = np.arange(epochs) + 1

        ax.plot(time, test_mean, label='CMIP5')
        ax.plot(time, pred_mean, label='LSTM')

        ax.set_ylabel('Global Mean Temperature (K)')
        ax.legend()
        ax.grid()

        plt.savefig(path + "LSTM_globmean_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (neurons, batch_size, epochs, time_steps), dpi=400)

        """

if __name__ == "__main__":
    #neurons = 50
    #epochs = 20
    #time_steps = 12
    #batch_size = int(64 / 4)

    # change working/data directory
    wdir = './'
    #wdir = "/home/mpim/m300517/Hausaufgaben/bdp_cnn/bdp_cnn/cmip5/"

    # implement run directory
    folder = '20180518_1317_42s'
    file = glob.glob('RMSE*.nc')
    path = "C:/Users/darkl/Desktop/runs/" + folder + '/'

    ev = Evaluater()
    dh = DataHandler()

    trues, preds, runtime, epochs, time_steps, batch_size, neurons = dh.get_results(file, path=path)

    #time, lat, lon = dh.get_dims("./data/lkm0401_echam6_BOT_mm_1850-2005.nc")

    ev.map_mae(trues, preds, neurons, batch_size, epochs, time_steps, runtime, path=path)

    #history = dh.get_history(path=path)
    #ev.model_loss(history['loss'], history['val_loss'],
    #                 neurons, batch_size, epochs, time_steps, runtime, path=path)
    #ev.model_lr(history['lr'],
    #              neurons, batch_size, epochs, time_steps, runtime, path=path)
