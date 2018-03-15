from bdp_cnn.Lorenz.NN_Lorenz import NN
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class CNN(NN):
    """
    Convolutional neural network (CNN) for timeseries prediction with using the Lorenz model.
    """

    def init_model(self, nb_filters=4, filter_size=(2, 1), grid_size=(40, 1), nb_ts=None):
        """
        Initialisation of CNN model

        Args:
            nb_filters (int): The number of different filters to learn.
            filter_size: The filter size
            grid_size:
            nb_ts:

        Returns:

        """

        """
        Initialisation of CNN model
        """

        # shape of input (time, gridx, gridy=1, 1)
        model = Sequential()
        # input_shape. tuple does not include the sample axis
        # padding. zero padding is activated, thus the output shape persits
        model.add(Conv2D(filters=nb_filters,
                         kernel_size=filter_size,
                         activation='relu',
                         padding='same',
                         input_shape=(grid_size[0], grid_size[1], 1)))

        model.compile(loss='mse', optimizer='adam')

        return model

    def predict(self):
        """
        Prediction with model

        """
        pass

    def scale(self, array):
        """
        Scale of input data
        """
        scaler = MinMaxScaler(feature_range=(-1, 1))
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

        train = array[int(4/6 * array.shape[0]), :, :]
        test = array[int(4/6 * array.shape[0]), :, :]
        val = array[int(4/6 * array.shape[0]), :, :]
        return train, test, val

    def test_cnn(self):
        model_test = self.init_model()
        data = self.read_netcdf("100_years_1_member.nc")

        x = data.copy()
        y = np.roll(data.copy(), 1, axis=0)

        xscaler, x = self.scale(x)
        yscaler, y = self.scale(y)

        xtrain, xtest, xval = self.split_ds(x)
        ytrain, ytest, yval = self.split_ds(y)



if __name__ == "__main__":
    cnn_model = CNN()
    cnn_model.test_cnn()
