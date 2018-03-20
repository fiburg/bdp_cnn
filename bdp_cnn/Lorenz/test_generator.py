from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense,LSTM

import numpy as np

length = 201
data = np.array([[i/length] for i in range(length)])
targets = np.array([[i/length] for i in range(length)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=1,
                               batch_size=10)

batch_0 = data_gen[0]
x, y = batch_0

model = Sequential()
model.add(LSTM(100,batch_size=10,stateful=False,input_shape=(10,1)))
model.add(Dense(1,activation="relu"))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

for i in range(12):
    print(i)
    model.fit_generator(data_gen,shuffle=False,epochs=1,verbose=1)
    model.reset_states()

trained_weights = model.get_weights()

pred_model = Sequential()
pred_model.add(LSTM(units=100,batch_size=1,stateful=False,input_shape=(10,1)))
pred_model.add(Dense(1,activation="relu"))
pred_model.set_weights(trained_weights)
pred_model.compile(loss='mean_squared_error', optimizer='adam')

test_x = np.array([[(i+10)/length] for i in range(10)])
test_y = np.array([[(i+10)/length] for i in range(11)])[-1]
test_x = test_x.reshape(1,10,1)
p = pred_model.predict(test_x)
print("Truth: %.5f   | Prediction: %.5f "%(test_y,p[0]))
