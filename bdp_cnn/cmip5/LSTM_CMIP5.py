"""
Class to work with the LSTM RNN. The class completely uses the keras API.
For an example to define a complete model run checkout the function autorun at the bottom of this module.
"""


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
        self.data_dim = 40
        self.time_steps=time_steps

    def getdata(self,file):
        from .datahandler import DataHandler

        _data = DataHandler.get_vamr(file,var_name="var167")

        self.data = DataHandler.shape(_data)




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
        f0 = 2/3 * len(self.data.value)+27
        f1 = 5/6 * len(self.data.value)+5

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


    def fit_model(self):
        """
        Fitting the model to the input-data and parameters. It will therefore use the generators.
        At the end the prediction-model will be initialized automatically and will replace the training model.

        """
        self.model.fit_generator(self.train_gen,shuffle=False,epochs=self.nb_epoch,
                                 validation_data=self.valid_gen,verbose=1)

        self.__init_pred_model()

    def __init_pred_model(self):
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
                            input_shape=(self.time_steps,self.data_dim))),
        self.model.add(Dense(self.data_dim))
        self.model.set_weights(weights)
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def evaluate(self):
        """
        Making predictions with the testing model using the testing-data-generator.

        Returns:
            tuple(py,preds)
                py: numpy array of target values (Tuth values)
                preds: numpy array of LSTM prediction for the targets
        """

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

    def analysis_scatter(self, ytest, ypred,runtime):
        """
        Creates a scatterplot plotting the truth vs. the prediction.
        Furthermore plots a regression line and writes some stats in the tilte.

        Args:
            truth: numpy array
            preds: numpy array

        Returns:

        """

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
    """
    Example function for a complete model run, saving the results as an image.

    """
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

    model.analysis_scatter(truth,preds,runtime)

if __name__ == "__main__":

    neurons = 50
    epochs = 1
    time_steps = 10
    batch_size = 5

    start = timeit.default_timer()
    model = LSTM_model(neurons=neurons, nb_epoch=epochs, time_steps=time_steps, batch_size=batch_size)
    model.getdata()
    model.createGenerators()
    model.init_model()
    model.fit_model()
    truth, preds = model.evaluate()
    truth = model.scale_invert(truth)
    preds = model.scale_invert(preds)
    stop = timeit.default_timer()
    runtime = stop-start

    model.analysis_scatter(truth,preds,runtime)




