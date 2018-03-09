from keras.models import Sequential
from keras.layers import Dense,MaxPool1D,Dropout,LSTM
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint

class CNN(object):

    def __init__(self, x=None, y=None, split=10000):
        self.model = None
        self.data_from_netcdf = x
        self.data = None  # data
        self.scaler = 1  # factor to scale
        self.test = None
        self.train = None
        self.train_scaled = None
        self.test_scaled = None
        self.supervised_values = None
        self.raw_values = None
        self.bach_size = 1
        self.split = -split

    def __str__(self):
        attr_list = [self.model, self.data_from_netcdf, self.data, self.scaler, self.test, self.train, self.train_scaled,
                     self.test_scaled, self.supervised_values, self.raw_values, self.bach_size, self.split]

        attr_names = ["model","data_from_netcdf","data","scaler","test","train","train_scaled",
                     "test_scaled", "supervised_values","raw_values","bach_size","split"]

        s1 = [str("Following attributes are defined at this stage:")]
        for name,attr in zip(attr_names,attr_list):
            if np.any(attr):
                s1.append(name + " : " + str(attr) )

        return "\n".join(s1)


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

        self.data_from_netcdf = x[0, :, :]  # pass only one gridpoint for now... [ensemble, time, grid]


    def init_lstm(self, batch_size=None, nb_epoch=5, neurons=20):
        """
        set up model (init, comp, fit/train)

        Args:
            batch_size:
            nb_epoch:
            neurons:

        Returns:

        """
        if not batch_size:
            batch_size = self.bach_size
        else:
            self.bach_size = batch_size
        X, y = self.train[:, 0:-1], self.train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        checkpoint = ModelCheckpoint("weights.{epoch:02d}.hdf5", monitor='val_loss',
                                     save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        callback_list = [checkpoint]
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False,callbacks=callback_list)
            model.reset_states()
        self.model = model
        self.train_reshaped = self.train_scaled[:, 0].reshape(len(self.train_scaled), 1, 1)

    def predict(self):

        self.model.predict(self.train_reshaped, batch_size=self.bach_size)

    def scale(self):
        # fit scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = self.scaler.fit(self.train)
        # transform train
        self.train = self.train.values.reshape(self.train.shape[0], self.train.shape[1],self.train.shape[2])
        self.train_scaled = self.scaler.transform(self.train)
        # transform test
        self.test = self.test.values.reshape(self.test.shape[0], self.test.shape[1],self.test.shape[2])
        self.test_scaled = self.scaler.transform(self.test)


    def invert_scale(self, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        return inverted[0, -1]

    def create_train_test(self):
        self.train, self.test = self.supervised_values[0:self.split,:], self.supervised_values[-self.split:,:]


    def walk_forward_validation(self):

        def forecast_lstm(model, batch_size, X):
            X = X.reshape(1, 1, len(X))
            yhat = model.predict(X, batch_size=batch_size)
            return yhat[0, 0]

        self.predictions = list()
        for i in range(len(self.test_scaled)):
            # make one-step forecast
            X, y = self.test_scaled[i, 0:-1], self.test_scaled[i, -1]
            yhat = forecast_lstm(self.model, 1, X)
            # invert scaling
            yhat = self.invert_scale(X, yhat)
            # store forecast
            self.predictions.append(yhat)
            # expected = self.raw_values[len(self.train) + i + 1]
            # print('Hour=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))


    def make_supervised(self, lag=1):
        """
        Takes a numpy array as input and creates a second array which lacks the input array behind.
        The array lagging behind will be used as input data for the cnn while the other will be
        considered the "Truth".

        Args:
            data: numpy-array (2D).
            lag: int: number of steps one array should lag behind the other.

        Returns:
            pandas dataframe containing the original and the lagging timeseries.
        """

        df = pandas.DataFrame(self.data_from_netcdf)
        self.raw_values = df.values
        df_shift = pandas.DataFrame(np.roll(self.data_from_netcdf, lag, axis=0))
        df = df.iloc[:, 0:40].apply(tuple, axis=1)
        df_shift = df_shift.iloc[:, 0:40].apply(tuple, axis=1)
        df = pandas.concat([df_shift, df], axis=1)
        df = df.iloc[1:]
        self.supervised_values = df

    def report_performance(self):

        rmse = np.sqrt(mean_squared_error(self.raw_values[-self.split:], self.predictions))
        print('Test RMSE: %.3f' % rmse)
        # line plot of observed vs predicted
        fig,ax = plt.subplots()
        ax.plot(self.raw_values[-self.split:],label="Truth")
        ax.plot(self.predictions,label="Machine learning")
        ax.legend(loc="upper left")
        plt.savefig("complete.png")

        fig,ax = plt.subplots()
        ax.plot(self.raw_values[-self.split:],label="Truth")
        ax.plot(self.predictions,label="Machine learning")
        ax.set_xlim(0,10000)
        ax.legend(loc="upper left")
        plt.savefig("last_10k.png")

        fig, ax = plt.subplots()
        ax.plot(self.raw_values[-self.split:],label="Truth")
        ax.plot(self.predictions,label="Machine learning")
        ax.set_xlim(0, 1000)
        ax.legend(loc="upper left")
        plt.savefig("last_1k.png")






if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()
    cnn = CNN()
    cnn.read_netcdf("100_years_1_member.nc")    # done
    cnn.make_supervised()   # finn todo
    #cnn.create_train_test()     # tobi todo
    #cnn.scale()
    #cnn.init_lstm(batch_size=1,nb_epoch=1,neurons=100)
    #cnn.predict()
    #cnn.walk_forward_validation()
    #cnn.report_performance()
    #stop = timeit.default_timer()
    #runtime = stop-start
    #print("Runtime: " )
    #print(runtime)


