from bdp_cnn.Lorenz.NN_Lorenz import NN
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class CNN(NN):
    """
    Convolutional neural network (CNN) for timeseries prediction with using the Lorenz model.
    """

    def __init__(self):
        self.model = None
        self.init_model()

    def init_model(self, nb_features=1, filter_size=(2, 1), grid_size=(40, 1)):
        """
        Initialisation of CNN model

        Args:
            nb_features (int): The number of different filters to learn.
            filter_size: The filter size
            grid_size: input shape
        """

        """
        Initialisation of CNN model
        """

        # shape of input (time, gridx, gridy=1, 1)
        self.model = Sequential()
        # input_shape. tuple does not include the sample axis
        # padding. zero padding is activated, thus the output shape persits
        self.model.add(Conv2D(filters=nb_features,
                         kernel_size=filter_size,
                         activation='relu',
                         padding='same',
                         input_shape=(grid_size[0], grid_size[1], 1)))

        #self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

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
        self.model.fit(x_train, y_train, nb_epoch=4, batch_size=32, validation_data=(x_val, y_val))

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
        print('Test RMSE: %.3f' % rmse)

        m, b = np.polyfit(ypred.reshape(shape), ytest.reshape(shape), 1)
        x = ypred.reshape(shape)
        y = np.add(np.multiply(m, x), b)

        fig, ax = plt.subplots()

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