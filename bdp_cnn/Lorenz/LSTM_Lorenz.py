from bdp_cnn.Lorenz.NN_Lorenz import NN
from bdp_cnn.Lorenz.scaling import scale
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import timeit


class LSTM_model(NN):

    def __init__(self,data=None,batch_size=None,nb_epoch=None,neurons=None,time_steps=None):
        self.data = data
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.neurons = neurons
        self.data_dim = 40
        self.time_steps=time_steps



    def init_model(self,batch_size=None,nb_epoch=None,neurons=None):
        # get arguments:
        if batch_size:
            self.batch_size = batch_size
        if nb_epoch:
            self.nb_epoch = nb_epoch
        if neurons:
            self.neurons = neurons



        assert self.neurons
        assert self.nb_epoch
        assert self.batch_size
        assert self.time_steps
        assert self.data_dim

        # setup model:

        self.model = Sequential()
        self.model.add(LSTM(units=self.neurons,
                            batch_size=self.batch_size,
                            # return_sequences=True,
                            stateful=False,
                            input_shape=(self.time_steps,self.data_dim))),
        self.model.add(Dense(self.data_dim))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self


    def createGenerators(self):

        f0 = 2/3 * len(self.data.value)+27
        f1 = 5/6 * len(self.data.value)+5

        self.train_gen = TimeseriesGenerator(
            self.data.value[:int(f0)], self.data.value[:int(f0)],
            sampling_rate=1,shuffle=False,
            length=self.time_steps, batch_size=self.batch_size
        )
        self.valid_gen = TimeseriesGenerator(
            self.data.value[int(f0):int(f1)], self.data.value[int(f0):int(f1)],
            sampling_rate=1,shuffle=False,
            length=self.time_steps, batch_size=self.batch_size
        )

        self.test_gen = TimeseriesGenerator(
            self.data.value[int(f1):], self.data.value[int(f1):],
            sampling_rate=1,shuffle=False,
            length=self.time_steps, batch_size=1
        )


    def fit_model(self):
        self.model.fit_generator(self.train_gen,shuffle=False,epochs=self.nb_epoch,
                                 validation_data=self.valid_gen,verbose=1)

        self.__init_pred_model()

    def __init_pred_model(self):
        self.training_model = self.model
        weights = self.training_model.get_weights()

        self.model = Sequential()
        self.model.add(LSTM(units=self.neurons,
                            batch_size=1,
                            # return_sequences=True,
                            stateful=False,
                            input_shape=(self.time_steps,self.data_dim))),
        self.model.add(Dense(self.data_dim))
        self.model.set_weights(weights)
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def walk_forward_validation(self):

        def forecast_lstm(model, batch_size, X):
            X = X.reshape(1,10, 40)
            yhat = model.predict(X, batch_size=batch_size)
            return yhat[0, :]

        self.predictions = []

        for i in range(len(self.x)):
            # make one-step forecast
            X = self.x[i:i+10,:]
            yhat = forecast_lstm(self.model, 100, X)
            self.yhat = yhat
            self.X = X
            # invert scaling
            # yhat = self.invert_scale(X.reshape(1,len(X)), yhat,self.test_scaler)
            # store forecast
            self.predictions.append(scale(is_already_dimensionless=True).T(yhat).invert().value)

    def evaluate(self):

        print("Evaluating the model...")
        py = np.zeros([len(self.test_gen),40])
        for i in range(len(self.test_gen)):
            py[i] = (self.test_gen[i][1][0][:])

        preds = self.model.predict_generator(self.test_gen)
        # print("Truth: %.5f   | Prediction: %.5f "%(test_y*scaler,p[0]*scaler))
        # pred_model.evaluate_generator(test_gen)


        return py,preds

    def predict(self,value):
        self.model.predict(value)


    def scale(self,var="T"):
        pass



    def scale_invert(self,value):
        ret = self.data.scaler.inverse_transform(value)

        return ret

    def present(self,truth,preds):
        print("plotting Results...")


        fig, ax = plt.subplots(figsize=(7,4))
        fig.suptitle("LSTM with %i neurons, %i batchsize,\n %i epochs and %i timesteps" %
                     (self.neurons, self.batch_size, self.nb_epoch, self.time_steps))

        x = truth[:,:]
        y = preds[:,:]

        # print("reshaping x and y...")
        # x = x.reshape(x.shape[0] * x.shape[1])
        # y = y.reshape(y.shape[0] * y.shape[1])


        ax.scatter(x[:,:],y[:,:],alpha=0.09)


        ax.set_xlabel("Lorenz-Truth Temperature [K]")
        ax.set_ylabel("LSTM-Prediction Temperature[K]")
        ax.set_xlim(260, 300)
        ax.set_ylim(260, 300)
        ax.plot(np.linspace(260,300,100),np.linspace(260,300,100),lw=1,color="black")
        # plt.legend(ncol=4,fancybox=True)


        print("\t saving figure...")
        plt.savefig("LSTM_%ineurons_%ibatchsize_%iepochs_%itimesteps.png"%
                    (self.neurons,self.batch_size,self.nb_epoch,self.time_steps),dpi=400)

    def analysis_scatter(self, ytest, ypred,runtime):
        print(ytest.shape)
        print(ypred.shape)

        shape = ypred.shape[0] * ypred.shape[1]

        rmse = np.sqrt(mean_squared_error(ytest.reshape(shape), ypred.reshape(shape)))
        corr = np.corrcoef(ytest.reshape(shape), ypred.reshape(shape))

        m, b = np.polyfit(ytest.reshape(shape), ypred.reshape(shape), 1)
        x = range(-22, 22, 1)
        yreg = np.add(np.multiply(m, x), b)

        print("plotting Results...")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(
            'LSTM with {0} neurons, {1} batchsize, {2} epochs and {3} timesteps\n RMSE = {4:.3f} '\
            'and CORR = {5:.3f}, runtume = {6:.2f} '.format(
                self.neurons,
                self.batch_size,
                self.nb_epoch,
                self.time_steps,
                rmse, corr[0, 1],runtime))
        ax.plot(ytest.reshape(shape), ypred.reshape(shape), lw=0, marker=".", color="blue", alpha=0.05,
                markeredgewidth=0.0)
        ax.plot(x, yreg, '-', label="Regression", color="red", lw=2)
        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xlabel("Test")
        ax.set_ylabel("Prediction")
        ax.set_xlim(-10, 20)
        ax.set_ylim(-10, 20)
        print("\t saving figure...")
        plt.savefig("Images/CNN_%ineurons_%ibatchsize_%iepochs_%itimesteps.png" %
                    (self.neurons, self.batch_size, self.nb_epoch, self.time_steps), dpi=400)




def autorun(neurons,epochs,time_steps,batch_size):
    start = timeit.default_timer()
    model = LSTM_model(neurons=neurons, nb_epoch=epochs, time_steps=time_steps, batch_size=batch_size)
    data = model.read_netcdf("100_years_1_member.nc") # temperatures in Kelvin instead of Celsius
    temp = scale().T(data)
    model.data = temp
    model.createGenerators()
    model.init_model()
    model.fit_model()
    truth, preds = model.evaluate()
    truth = model.scale_invert(truth)
    preds = model.scale_invert(preds)
    stop = timeit.default_timer()
    runtime = stop-start

    # model.present(truth, preds)
    model.analysis_scatter(truth,preds,runtime)

if __name__ == "__main__":

    # for time_steps in [2,5,10,20,50,100]:
        for neurons in np.arange(50,300,50):
            for epochs in [1,5,10]:
                autorun(neurons=neurons,time_steps=10,epochs=epochs,batch_size=50)




