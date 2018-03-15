from bdp_cnn.Lorenz.NN_Lorenz import NN
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import Sequential


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

    def scale(self):
        """
        Scale of input data
        """

    def scale_invert(self):
        """
        Invert application of `scale`
        """

    def test_cnn(self):
        model_test = self.init_model()


if __name__ == "__main__":
    cnn_model = CNN()
    cnn_model.test_cnn()
