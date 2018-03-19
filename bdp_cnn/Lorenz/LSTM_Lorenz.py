from bdp_cnn.Lorenz.NN_Lorenz import NN
from bdp_cnn.Lorenz.scaling import scale
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,LSTM


class LSTM_class(NN):

    def __init__(self,data=None,batch_size=None,nb_epoch=None,neurons=None):
        self.data = data
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.neurons = neurons

    def batch_size_avail(self):

        return list(np.where(np.mod([x for x in range(len(self.data))],self.batch_size) == 0))



    def init_model(self,time_steps,auto_batch_size=True,batch_size=None,nb_epoch=None,neurons=None):
        # get arguments:
        if batch_size:
            self.batch_size = batch_size
        if nb_epoch:
            self.nb_epoch = nb_epoch
        if neurons:
            self.neurons = neurons

        self.time_steps = time_steps

        if auto_batch_size:
            self.batch_size = self.data.value.shape[0]

        # setup model:

        self.model = Sequential()
        self.model.add(LSTM(batch_size=self.batch_size,
                            return_sequences=True,
                            stateful=True,
                            batch_input_shape=(self.batch_size,self.time_steps, self.data.value.ndim)))
        self.model.add(Dense(40))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self

    def reshape_data(self):
        self.x = self.data.value.r


    def train_model(self):
        pass




    def predict(self):
        pass


    def scale(self,var="T"):
        pass



    def scale_invert(self):
        pass



if __name__ == "__main__":

    model = LSTM_class(neurons=100,nb_epoch=2)
    data = np.add(model.read_netcdf("100_times_1000_days.nc"),273.15) # temperatures in Kelvin instead of Celsius
    temp = scale().T(data)
    model.data = temp
    model.init_model(time_steps=10)




