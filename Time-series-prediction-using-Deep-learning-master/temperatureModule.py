
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

df1 = pd.read_csv("mpi_roof_2016b.zip", sep="\,\s*", encoding='cp1252')
print(df1.head())

dfs = []
for j in range(2009, 2017):
    for i in ['a', 'b']:
        _df = pd.read_csv("mpi_roof_{0}{1}.zip".format(j, i),
                          sep="\,\s*", encoding='cp1252')
        dfs.append(_df)
        print(j, i, _df.shape)

df = pd.concat(dfs)
print(df[::720].head(10))
print(df.shape)
df.set_index('"Date Time"', inplace=True)
print(df.head(2))
plt.figure()
plt.plot(pd.to_datetime(df.index[::720]))
plt.show()
plt.figure()
plt.plot(pd.to_datetime(df.index[::720], dayfirst=True, infer_datetime_format=True))
plt.show()
df.index=pd.to_datetime(df.index, dayfirst=True, infer_datetime_format=True)
float_data = df.values
pd.options.display.max_columns=50
print(df.describe())
df[[df.columns[1]]].plot(ylim=[-20,40], style='.', markersize=1)
plt.show()
df[[df.columns[1]]][:1400].plot(style='.')
plt.show()

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=10,
                              validation_data=val_gen,
                              validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
print(history.history)
epochs = range(len(loss))
model.save('temperature_model.h5')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
