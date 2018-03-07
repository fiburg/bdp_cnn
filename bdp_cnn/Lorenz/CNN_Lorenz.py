from keras.models import Sequential
from keras.layers import Dense,MaxPool1D,Dropout
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


class CNN(object):

    def __init__(self, x=None, y=None):
        self.model = None
        self.x_train = x
        self.y_train = y

    def run_model(self):
        if ((not self.x_train) or (not self.y_train)):
            print("x_train or y_train not set.")
            return None

        self.model = Sequential()
        self.model.add(Dense(units=1, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)

    def get_keys(self,file_name):
        nc = Dataset(file_name)
        keylist = []
        for key in nc.variables.keys():
            if ((not key == "time") and (not key == "grid")):
                keylist.append(key)
        return keylist

    def read_netcdf(self,file_name,keys=None):
        if not keys:
            keys = self.get_keys(file_name)

        nc = Dataset(file_name)
        dim1,dim2 = np.shape(nc.variables[keys[0]])
        x  = np.zeros([len(keys),dim1-1,dim2])
        y = np.asarray(x.copy())
        for i,key in enumerate(keys):
            x_tmp = nc.variables[key][:].copy()
            y_tmp = x_tmp[1:,:]
            x_tmp = x_tmp[:-1,:]
            x[i,:,:] = x_tmp
            y[i,:,:] = y_tmp

        self.x_train = x
        self.y_train = y





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
    cnn.read_netcdf("test_file.nc")

