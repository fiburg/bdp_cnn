from keras.models import Sequential
from keras.layers import Dense,MaxPool1D,Dropout
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas
from sklearn.preprocessing import MinMaxScaler

class CNN(object):

    def __init__(self, x=None, y=None):
        self.model = None
        self.x_train = x
        self.y_train = y
        self.data = None

    def run_model(self):

        self.model = Sequential()
        self.model.add(Dense(units=3, activation='relu', input_dim=np.ndim(self.x_train)))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(units=1, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)

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

    def read_netcdf(self,file_name,keys=None):
        """
        Reads data from netcdf and stores it to x_train and y_train.

        Args:
            file_name: str: name and path to file.
            keys: list: optional, which keys to use. If None, then all keys will be used.

        """
        if not keys:
            keys = self.get_keys(file_name)

        nc = Dataset(file_name)
        dim1,dim2 = np.shape(nc.variables[keys[0]])
        x  = np.zeros([len(keys),dim1,dim2])
        for i,key in enumerate(keys):
            x_tmp = nc.variables[key][:].copy()
            x_tmp = x_tmp[:,:]
            x[i,:,:] = x_tmp


        # for x_i,y_i in zip(x,y):
        #     if not "x_new" in locals():
        #         x_new = x_i
        #         y_new = y_i
        #     else:
        #         x_new = np.concatenate((x_new,x_i))
        #         y_new = np.concatenate((y_new,y_i))

        self.x_train = x[0,:,0] # pass only one gridpoint for now

    def prepare_data_1_member(self):
        self.data = timeseries_to_supervised(self.x_train)
        X = self.data.values[:,1]
        X = X.reshape(len(X), 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X)
        scaled_X = scaler.transform(X)
        scaled_series = pandas.Series(scaled_X[:, 0])
        print(scaled_series.head())
        # invert transform
        inverted_X = scaler.inverse_transform(scaled_X)
        inverted_series = pandas.Series(inverted_X[:, 0])
        print(inverted_series.head())





def timeseries_to_supervised(data, lag=1):
    """
    Takes a numpy array (1D!!!) as input and creates a second array which lacks the input array behind.
    The array lagging behind will be used as input data for the cnn while the other will be
    considered the "Truth".

    Args:
        data: numpy-array (2D).
        lag: int: number of steps one array should lag behind the other.

    Returns:
        pandas dataframe containing the original and the lagging timeseries.
    """

    df = pandas.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pandas.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pandas.Series(diff)

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

if __name__ == "__main__":
    from bdp_cnn.Lorenz.DataCreation1 import creation_main
    from bdp_cnn.Lorenz.LorenzDataCreation import Lorenz

    # x = np.random.rand(int(1e6))
    # x = np.multiply(x,100)
    # y = np.divide(x.copy(), 2)
    # cnn = CNN(x, y)
    # test = np.random.rand(100)
    # truth = np.divide(test,2)
    # tested = cnn.model.predict(test)
    # results = np.subtract(tested[:,0], truth)
    # plt.plot(results)


    cnn = CNN()
    cnn.read_netcdf("100_years_1_member.nc")
    cnn.prepare_data_1_member()


