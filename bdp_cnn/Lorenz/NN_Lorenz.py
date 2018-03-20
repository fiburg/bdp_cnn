import abc
from netCDF4 import Dataset
import numpy as np


class NN(abc.ABC):
    """
    Abstract class for neural network implemention like CNN or LSTM.
    """

    def get_keys(self,file_name):
        """
        Get a list of all keys, excluding "time" and "grid"

        Args:
            file_name: str: name and path to file.

        """

        nc = Dataset(file_name)
        keylist = []
        for key in nc.variables.keys():
            if ((not key == "time") and (not key == "grid")):
                keylist.append(key)

        nc.close()
        return keylist

    def read_netcdf(self, file_name, keys=None):
        """
        Reads data from netcdf and stores it to x_train and y_train.

        Args:
            file_name: str: name and path to file.
            keys: list: optional, which keys to use. If None, then all keys will be used.

        """
        if not keys:
            keys = self.get_keys(file_name)

        nc = Dataset(file_name)
        dim1, dim2 = np.shape(nc.variables[keys[0]])
        x = np.zeros([len(keys), dim1, dim2])
        for i, key in enumerate(keys):
            x_tmp = nc.variables[key][:].copy()
            x_tmp = x_tmp[:, :]
            x[i, :, :] = x_tmp

        return x[0, :, :]

    @abc.abstractmethod
    def init_model(self):
        """
        Initialisation of neural network model
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """
        Prediction with model

        """
        pass

    @abc.abstractmethod
    def scale(self):
        """
        Scale of input data
        """

    @abc.abstractmethod
    def scale_invert(self):
        """
        Invert application of `scale`
        """