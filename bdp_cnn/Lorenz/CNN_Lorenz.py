from keras.models import Sequential
from keras.layers import Dense,MaxPool1D,Dropout
import numpy as np
import matplotlib.pyplot as plt


class CNN(object):

    def __init__(self, x, y):
        self.model = None
        self.x_train = x
        self.y_train = y

        self.__createModel()

    def __createModel(self):
        self.model = Sequential()
        self.model.add(Dense(units=1, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1, activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)


if __name__ == "__main__":
    from bdp_cnn.Lorenz.DataCreation1 import creation_main
    from bdp_cnn.Lorenz.LorenzDataCreation import Lorenz

    x = np.random.rand(int(1e6))
    x = np.multiply(x,100)
    y = np.divide(x.copy(), 2)
    cnn = CNN(x, y)
    test = np.random.rand(100)
    truth = np.divide(test,2)
    tested = cnn.model.predict(test)
    results = np.subtract(tested[:,0], truth)
    plt.plot(results)

