from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense,LSTM

import numpy as np
import matplotlib.pyplot as plt

length = 2022
scaler = 1000
# data = np.array([[i/scaler] for i in range(length)])
# targets = np.array([[i/scaler] for i in range(length)])

data = np.zeros([length,40]).astype(float)
for i in range(length):
    data[i,:] = np.multiply(np.sin(np.rad2deg(i)),np.add(np.random.rand(40)/10,1))

targets = data.copy()

data_gen = TimeseriesGenerator(data[:1511], targets[:1511],
                               length=10, sampling_rate=1,
                               batch_size=20)

batch_0 = data_gen[0]
x, y = batch_0

model = Sequential()
model.add(LSTM(100,batch_size=20,stateful=False,input_shape=(10,40)))
model.add(Dense(40))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

valid_gen = TimeseriesGenerator(data[1511:], targets[1511:],
                               length=10, sampling_rate=1,
                               batch_size=20)

model.fit_generator(data_gen,shuffle=False,epochs=10,verbose=1,validation_data=valid_gen)


trained_weights = model.get_weights()

pred_model = Sequential()
pred_model.add(LSTM(units=100,batch_size=1,stateful=False,input_shape=(10,40)))
pred_model.add(Dense(40))
pred_model.set_weights(trained_weights)
pred_model.compile(loss='mean_squared_error', optimizer='adam')

test_gen = TimeseriesGenerator(data[:1500],targets[:1500],length=10,sampling_rate=1,batch_size=1)

preds = pred_model.predict_generator(test_gen)
# print("Truth: %.5f   | Prediction: %.5f "%(test_y*scaler,p[0]*scaler))
# pred_model.evaluate_generator(test_gen)

py = []
for i in range(1489):
    py.append(test_gen[i][1][0][0])

fig,ax = plt.subplots()
ax.plot(py,preds)
ax.set_xlabel("Truth")
ax.set_ylabel("Prediction")
ax.set_title("Model predicting 2D-sinus-field with noise.")
plt.show()