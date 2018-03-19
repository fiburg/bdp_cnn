from bdp_cnn.Lorenz.NN_Lorenz import NN
from bdp_cnn.Lorenz.scaling import scale
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,LSTM


class LSTM_model(NN):

    def __init__(self,data=None,batch_size=None,nb_epoch=None,neurons=None):
        self.data = data
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.neurons = neurons
        self.data_dim = 40

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


        self.model.add(LSTM(units=self.batch_size,
                            batch_size=self.batch_size,
                            # return_sequences=True,
                            stateful=True,
                            batch_input_shape=(self.batch_size,self.time_steps, self.data_dim)))
                            # input_shape=(10,self.data_dim)))
        self.model.add(Dense(self.data_dim,activation="relu"))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self

    def reshape_data(self):
        self.x = self.data.value


    def train_model(self):

        for i in np.arange(0,1000,step=self.time_steps):
            self.model.train_on_batch(
                self.x[:,i:i+self.time_steps,:],
                np.reshape(self.x[:,i+self.time_steps,:],(self.batch_size,self.data_dim)))


    def walk_forward_validation(self):

        def forecast_lstm(model, batch_size, X):
            X = X.reshape(100, 10, 40)
            yhat = model.predict(X, batch_size=batch_size)
            return yhat[0, :]

        self.predictions = []

        for i in range(len(self.x)):
            # make one-step forecast
            X = self.x[:,i:i+10,:]
            yhat = forecast_lstm(self.model, 100, X)
            self.yhat = yhat
            self.X = X
            # invert scaling
            # yhat = self.invert_scale(X.reshape(1,len(X)), yhat,self.test_scaler)
            # store forecast
            self.predictions.append(scale(is_already_dimensionless=True).T(yhat).invert().value)



    def predict(self):
        model.predict()


    def scale(self,var="T"):
        pass



    def scale_invert(self):
        pass



if __name__ == "__main__":

    model = LSTM_model(neurons=100, nb_epoch=2)
    data = np.add(model.read_netcdf("100_times_1000_days.nc"),273.15) # temperatures in Kelvin instead of Celsius
    temp = scale().T(data)
    model.data = temp
    model.init_model(time_steps=10)
    model.reshape_data()
    model.train_model()
    model.walk_forward_validation()




