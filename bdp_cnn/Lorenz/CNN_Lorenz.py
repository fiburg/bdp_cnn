from bdp_cnn.Lorenz.NN_Lorenz import NN
from keras.layers import Conv2D, Input, Dense, InputLayer
from keras.models import Model
from keras.callbacks import TensorBoard
from tensorflow import pad, constant
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class CNN(NN):
    """
    Convolutional neural network (CNN) for timeseries prediction with using the Lorenz model.
    """

    def __init__(self, epochs=3, batch=64, filter_length=2):
        self.model = None
        self.epochs = epochs
        self.batch = batch
        self.filter = filter_length
        self.init_model()

    def init_model(self, nb_filters=1, filter_size=(5, 1), grid_size=(40, 1)):
        """
        Initialisation of CNN model

        Args:
            nb_filters (int): The number of different filters to learn.
            filter_size: The filter size
            grid_size: shape of input (time, gridx, gridy=1, 1)
        """

        inputs = Input(shape=(grid_size[0], grid_size[1], 1))

        conv2d_first = Conv2D(filters=nb_filters,
                         kernel_size=filter_size,
                         activation='relu')(inputs)

        paddings = constant([[0, 0], [2, 2], [0, 0], [0, 0]])
        # currently zero padding, replace CONSTANT with REFLECT
        padding_ref = InputLayer(input_tensor=pad(conv2d_first, paddings, "CONSTANT"))

        # Tensorflow tensor is not accepted... Output tensors to a Model must be Keras tensors

        # input_shape. tuple does not include the sample axis
        self.model = Model(inputs=inputs, outputs=padding_ref)

        opt = optimizers.Adadelta()

        self.model.compile(loss='mse', optimizer=opt, metrics=['acc', 'mae'])

    def predict(self, x_test):
        """
        Prediction with model

        """
        yhat = self.model.predict(x_test)

        return yhat

    def fit(self, x_train, y_train, x_val, y_val):
        """
        Train the model

        """

        tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0,
                                  write_graph=True, write_images=True)
        callbacks = []
        callbacks.append(tb_callback)

        self.model.fit(x_train, y_train,
                       epochs=self.epochs,
                       batch_size=self.batch,
                       validation_data=(x_val, y_val),
                       callbacks=callbacks)

    def scale(self, array):
        """
        Scale of input data
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(array)
        return scaler, scaler.transform(array)

    def scale_invert(self, scaler, array):
        """
        Invert application of `scale`
        """
        pass

    def split_ds(self, array):
        """
        Split some dataset into training, testing and validation.

        Args:
            array: input data

        Returns:
            train, test, val

        """

        train = array[:int(4/6 * array.shape[0]), :, :, :]
        test = array[int(4/6 * array.shape[0]):int(5/6 * array.shape[0]), :, :, :]
        val = array[int(5/6 * array.shape[0]):, :, :, :]

        return train, test, val

    def analysis_scatter(self, ytest, ypred):
        shape = ypred.shape[0] * ypred.shape[1]

        rmse = np.sqrt(mean_squared_error(ytest.reshape(shape), ypred.reshape(shape)))

        m, b = np.polyfit(ypred.reshape(shape), ytest.reshape(shape), 1)
        x = ypred.reshape(shape)
        y = np.add(np.multiply(m, x), b)

        fig, ax = plt.subplots()
        plt.title('CNN, filter size {0:2d}, epochs {1:2d}, batch size {2:2d}, RMSE {3:4.3f}, zero padding'.format(self.filter,
                                                                                                                 self.epochs,
                                                                                                                 self.batch,
                                                                                                                 rmse))
        ax.plot(ypred.reshape(shape), ytest.reshape(shape), lw=0, marker=".", color="blue")
        ax.plot(x, y, '-', label="Regression", color="red", lw=2)
        ax.legend(loc="upper left")
        ax.grid()
        ax.set_ylabel("Test")
        ax.set_xlabel("Prediction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig("CNN_scatter.png")

if __name__ == "__main__":
    cnn = CNN()

    # data handling
    data = cnn.read_netcdf("100_years_1_member.nc")

    x = data.copy()
    y = np.roll(data.copy(), 1, axis=0)

    xscaler, x = cnn.scale(x)
    yscaler, y = cnn.scale(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1, 1))

    xtrain, xtest, xval = cnn.split_ds(x)
    ytrain, ytest, yval = cnn.split_ds(y)

    # train model

    cnn.fit(xtrain, ytrain, xval, yval)

    # prediction
    ypred = cnn.predict(xtest)

    # performance analysis
    cnn.analysis_scatter(ytest, ypred)

    cnn.model.summary()