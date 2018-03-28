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
        newtime = todate(time.astype('int').astype('str'), '%Y%m%d')

        nc.close()

        return time, lat, lon


if __name__ == "__main__":
    dh = DataHandler()

    #test = dh.get_var('./data/lkm0401_echam6_BOT_mm_1850-2005.nc', 'var167')
    #dh.get_dims('./data/lkm0401_echam6_BOT_mm_1850-2005.nc')