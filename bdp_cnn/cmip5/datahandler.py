from netCDF4 import Dataset
import numpy as np
from datetime import datetime

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

        todate = np.vectorize(datetime.strptime)
        newtime = todate(time.astype('str'), '%Y%m%d')
        print(newtime)

        nc.close()

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


if __name__ == "__main__":
    dh = DataHandler()

    #test = dh.get_var('./data/lkm0401_echam6_BOT_mm_1850-2005.nc', 'var167')
    dh.get_dims('./data/lkm0401_echam6_BOT_mm_1850-2005.nc')