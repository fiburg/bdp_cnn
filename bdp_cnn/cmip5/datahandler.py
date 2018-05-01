from netCDF4 import Dataset
import csv
import numpy as np
from datetime import datetime as dt
import collections
import time
import os

class DataHandler(object):
    """
    Class provides methods to handle data
    """

    def get_var(self, file_name, var_name):
        """
        Reads data/variable from netcdf cmip5 file.

        Args:
            file_name: str: name and path to file.
            var_name: str: name of the variable.

        Returns:
            var: numpy.ndarray: variable

        """
        nc = Dataset(file_name)

        var = nc.variables[var_name][:]

        nc.close()

        return var

    def get_dims(self, file_name):
        """

        Args:
            file_name: str: name and path to file.

        Returns:

        """
        nc = Dataset(file_name)

        time = nc.variables['time'][:]
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

        todate = np.vectorize(dt.strptime)
        newtime = todate(time.astype('int').astype('str'), '%Y%m%d')

        nc.close()

        return time, lat, lon

    def shape(self,array,inverse=False):
        """
        reshapes an array from 3D to 2D. If inverse==True then reshaping is from 2D to 3D using standard
        lat=192 and lon=96.

        Args:
            array: numpy array
            inverse: bool

        Returns:
            reshaped numpy array
        """

        #set data shape:
        lat = 192
        lon = 96

        if not inverse:
            return np.reshape(array,(array.shape[0],array.shape[1]*array.shape[2]))

        else:
            return np.reshape(array,(array.shape[0],lat,lon))

    def save_results(self,trues,preds,rmse,corr,runtime,neurons,epochs, time_steps,batch_size, file=None,path=''):
        """
        saves the lstm results to netcdf.

        Args:
            trues: numpy array: true values
            preds: numpy array: predicted values
            rmse: float: root mean squared error
            corr: float: correlation
            runtime: float: lstm runtime
            file: str: path of output file
            path: str: path of output folder
        """

        if path != '' and not os.path.exists(path):
            os.mkdir(path)

        if not file:
            now = dt.now().strftime("%Y%m%d_%H%M_%Ss")
            file = "RMSE%.2f_%s.nc"%(rmse,now)

        file = path + file

        nc = Dataset(file, mode="w")

        nc.creation_date = time.asctime()
        nc.RMSE = str(round(rmse,5))
        nc.CORR = str(round(corr,5))
        nc.Runtime = str(round(runtime,2))
        nc.epochs = str(epochs)
        nc.timesteps = str(time_steps)
        nc.batchsize = str(batch_size)
        nc.neurons = str(neurons)

        nc.createDimension("time",trues.shape[0])
        nc.createDimension("lat",trues.shape[1])
        nc.createDimension("lon",trues.shape[2])

        # time_var = nc.createVariable("time","f8",("time"))
        true_var = nc.createVariable("true_values","f8",("time","lat","lon"))
        true_var.description = "Values from the CMIP5 dataset which where used as testing data."

        pred_var = nc.createVariable("predictions","f8",("time","lat","lon"))
        pred_var.description = "From LSTM predicted values."

        true_var[:,:,:] = trues
        pred_var[:,:,:] = preds

        nc.close()

    def get_results(self, file, path=''):
        """
        reads the results of the lstm model netcdf output.

        Args:
            file: str: filename
            path: str: file path

        Returns:
            trues: numpy array: true values
            preds: numpy array: predicted values
            rmse: float: root mean squared error
            corr: float: correlation
            runtime: float: lstm runtime
        """

        nc = Dataset(path+file, mode='r')

        preds = nc.variables['predictions'][:]
        trues = nc.variables['true_values'][:]

        runtime = nc.Runtime
        rmse = nc.RMSE
        corr = nc.CORR

        nc.close()

        return trues, preds, runtime, rmse, corr

    def save_model(self, model, path=''):
        """
        Saves the model as json file with the trained weights in same folder.

            Args:
                folder: name of folder where model will be saved. (Default:./Date)

        """
        json_model = model.to_json()

        if path != '' and not os.path.exists(path):
            os.mkdir(path)

        with open(path + "model.json", "w") as f:
            f.write(json_model)

        model.save_weights(path + "weights.h5")

    def get_history(self, file='training.log', path=''):

        keys = []
        epochs = []
        loss = []
        lr = []
        val_loss = []

        with open(path+file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if not keys:
                    keys = row
                else:
                    epochs.append(int(row[0]))
                    loss.append(float(row[1]))
                    lr.append(float(row[2]))
                    val_loss.append(float(row[3]))

        history = {keys[0]: epochs,
                   keys[1]: loss,
                   keys[2]: lr,
                   keys[3]: val_loss}

        return history

if __name__ == "__main__":
    dh = DataHandler()
    #dh.get_results("RMSE7.09_20180424_1059_24s.nc")
    #dh.get_history()
    #test = dh.get_var('./data/lkm0401_echam6_BOT_mm_1850-2005.nc', 'var167')
    #dh.get_dims('./data/lkm0401_echam6_BOT_mm_1850-2005.nc')