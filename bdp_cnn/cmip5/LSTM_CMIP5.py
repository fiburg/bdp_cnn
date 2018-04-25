"""
Class to work with the LSTM RNN. The class completely uses the keras API.
For an example to define a complete model run checkout the function autorun at the bottom of this module.
"""

import matplotlib
matplotlib.use("Agg")
from bdp_cnn.cmip5.datahandler import DataHandler

from bdp_cnn.Lorenz.NN_Lorenz import NN
from bdp_cnn.Lorenz.scaling import scale
from bdp_cnn.cmip5.evaluater import Evaluater
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras import optimizers
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ReduceLROnPlateau, History

from sklearn.metrics import mean_squared_error
from datetime import datetime as dt
import timeit
import glob
import os

class LSTM_model(NN):
    """
    Class for setting up an LSTM. The adjustable parameters are:
     - input data, which will be splitted automatically into training, validation and testing datasets.
     - batch size, which affects model trainings speed and accuracy
     - number of training epochs
     - number of neurons
     - number of timesteps between which state is kept.

     While training, the Model uses evaluation data at the end of each epoch to evaluate itself, which results in much
     faster training.

     Furthermore there are actual two models being used. One for training and one for testing.
     The training model has a different batchsize for faster training.
     The testing model will always have a batchsize of 1 as you usually want to make predictions from one timeseries
     at the time.

    """

    def __init__(self,data=None,batch_size=None,nb_epoch=None,neurons=None,time_steps=None):
        """
        Initialization of the LSTM class.

        Args:
            data: str: path to netcdf-file
            batch_size: int: number of how many samples are trained at same time
            nb_epoch: int: number of epochs
            neurons: int: number of neurons
            time_steps: int: number of timesteps used for training AND prediction
        """

        self.data = data
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.neurons = neurons
        self.data_dim = 18432
        self.time_steps = time_steps
        self.history = None

    def getdata(self, file):
        """
        loads and scales data with ``DataHandler`` and ``scale``

        Args:
            file: str: file name
        """
        f0 = (4 * 1.5) * self.batch_size * self.time_steps + self.time_steps + 1
        f1 = f0 + (4 * 0.25) * self.batch_size * self.time_steps + self.time_steps + 1
        print(f0)
        _data = DataHandler().get_var(file ,var_name="var167")
        _data = DataHandler().shape(_data)
        _data_data = scale().T(_data[:int(f0)])
        _valid_data = scale().T(_data[int(f0):int(f1)])
        _test_data = scale().T(_data[int(f1):])


        if self.data == None:
            self.data = _data_data
            self.valid_data = _valid_data
            self.test_data = _test_data
        else:
            self.data.value = np.concatenate((self.data.value,_data_data.value),axis=0)

    def init_model(self,batch_size=None,nb_epoch=None,neurons=None):
        """
        Initializes the LSTM model. If not set already, batch_size, epochs and neurons can be set or reset.

        Args:
            batch_size: int: number of how many samples are trained at same time
            nb_epoch: int: number of epochs
            neurons: int: number of neurons

        Returns:

        """
        # get arguments:
        if batch_size:
            self.batch_size = batch_size
        if nb_epoch:
            self.nb_epoch = nb_epoch
        if neurons:
            self.neurons = neurons


        # make sure everything is set before continuing
        assert self.neurons
        assert self.nb_epoch
        assert self.batch_size
        assert self.time_steps
        assert self.data_dim

        # setup model:
        self.model = Sequential()
        self.model.add(LSTM(units=self.neurons,
                            batch_size=self.batch_size,
                            stateful=False,  # within one batch state is still kept. This just means between batches
                            input_shape=(self.time_steps,self.data_dim))),
        self.model.add(Dense(self.data_dim)) # <- does not really do much. Is just for output in right shape.
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        #self.model.compile(loss='mean_squared_error', optimizer=optimizers.Adadelta())
        return self


    def createGenerators(self):
        """
        Creates generators for the training, evaluation and testing data as they make life here a whole lot easier.
        The input is split here into training (2/3 of input data), evaluation data (1/6) and testing data (1/6.)


        Returns:
            sets the class variables:
                - train_gen: Generator for training data
                - valid_gen: Generator for validation data
                - test_gen: Generator for testing data

        """
        #f0 = 64*12+12+1

        f0 = (4*1.5) * self.batch_size*self.time_steps+self.time_steps+1

        f1 = f0 + (4*0.25) * self.batch_size * self.time_steps + self.time_steps + 1

        print(f0)
        print(f1)
        print(len(self.data.value))

        print(self.data.value[int(f1):int(f1)+12].shape)

        self.train_gen = TimeseriesGenerator(
            self.data.value[:int(f0)], self.data.value[:int(f0)],
            sampling_rate=1,shuffle=False, #shuffle=False is very important as we are dealing with continous timeseries
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

    def create_ensemble_generator(self):
        self.train_gen = TimeseriesGenerator(
            self.data.value, self.data.value,
            sampling_rate=1,shuffle=True, #shuffle=False is very important as we are dealing with continous timeseries
            length=self.time_steps, batch_size=self.batch_size
        )
        self.valid_gen = TimeseriesGenerator(
            self.valid_data.value, self.valid_data.value,
            sampling_rate=1,shuffle=True,
            length=self.time_steps, batch_size=self.batch_size
        )

        self.test_gen = TimeseriesGenerator(
            self.test_data.value, self.test_data.value,
            sampling_rate=1,shuffle=False,
            length=self.time_steps, batch_size=1
        )

    def fit_model(self):
        """
        Fitting the model to the input-data and parameters. It will therefore use the generators.
        At the end the prediction-model will be initialized automatically and will replace the training model.

        """

        # tensorboard not possible to use with validation generator in current keras version
        #tb_callback = TensorBoard(...)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5)

        # history callback returns loss and validation loss for each epoch
        history = History()

        callbacks = []
        #callbacks.append(tb_callback)
        callbacks.append(reduce_lr)
        callbacks.append(history)

        self.model.fit_generator(self.train_gen,shuffle=False,epochs=self.nb_epoch,
                                 validation_data=self.valid_gen, verbose=1,
                                 callbacks=callbacks)

        self.history = history.history

    def init_pred_model(self):
        """
        This function will init a new model for the prediction with the already trained weights.
        The new model is exactly the same as the old one, with only the batch-size differing.


        """
        self.training_model = self.model
        weights = self.training_model.get_weights()

        self.model = Sequential()
        self.model.add(LSTM(units=self.neurons,
                            batch_size=1,
                            # return_sequences=True,
                            stateful=False,
                            input_shape=(self.time_steps, self.data_dim))),
        self.model.add(Dense(self.data_dim))
        self.model.set_weights(weights)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        #self.model.compile(loss='mean_squared_error', optimizer=optimizers.Adadelta())


    def evaluate(self):
        """
        Making predictions with the testing model using the testing-data-generator.

        Returns:
            tuple(py,preds)
                py: numpy array of target values (Tuth values)
                preds: numpy array of LSTM prediction for the targets
        """

        print("Evaluating the model...")
        py = np.zeros([len(self.test_gen), self.data_dim])
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


if __name__ == "__main__":

    dh = DataHandler()
    ev = Evaluater()

    neurons = 50
    epochs = 50
    time_steps = 12
    batch_size = int(64 / 4)

    # change working/data directory
    wdir = './'
    #wdir = "/home/mpim/m300517/Hausaufgaben/bdp_cnn/bdp_cnn/cmip5/"

    datafolder = glob.glob(wdir + "data/*")
    print(datafolder)
    start = timeit.default_timer()
    model = LSTM_model(neurons=neurons, nb_epoch=epochs, time_steps=time_steps, batch_size=batch_size)
    model.init_model()

    for file in datafolder:
        model.getdata(file)

    model.create_ensemble_generator()
    model.fit_model()

    model.init_pred_model()
    truth, preds = model.evaluate()
    truth = model.scale_invert(truth)
    preds = model.scale_invert(preds)
    stop = timeit.default_timer()
    runtime = stop-start
    truth = dh.shape(truth,inverse=True)
    preds = dh.shape(preds,inverse=True)

    # OUTPUT
    folder = dt.now().strftime("%Y%m%d_%H%M_%Ss/")
    path = wdir + 'runs/' + folder

    if not os.path.exists(wdir + 'runs/'):
        os.mkdir(wdir + 'runs/')

    # save the model with results
    dh.save_model(model.model, path=path)

    corr = ev.calc_corr(truth, preds)
    rmse = ev.calc_rmse(truth, preds)
    dh.save_results(truth, preds, rmse, corr, runtime, path=path)

    # evaluate the model and plot the results
    ev.hist2d(truth, preds, neurons, batch_size, epochs, time_steps, runtime, path=path)
    ev.map_mae(truth, preds, neurons, batch_size, epochs, time_steps, runtime, path=path)

    ev.model_history(model.history['loss'], model.history['val_loss'],
                     neurons, batch_size, epochs, time_steps, runtime, path=path)
