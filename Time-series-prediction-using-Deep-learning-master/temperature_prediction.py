
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import layers
from keras.optimizers import RMSprop
dfs = []
for j in range(2009, 2017):
    for i in ['a', 'b']:
        _df = pd.read_csv("mpi_roof_{0}{1}.zip".format(j, i),
                          sep="\,\s*", encoding='cp1252')
        dfs.append(_df)
df = pd.concat(dfs)
df.set_index('"Date Time"', inplace=True)
df.index=pd.to_datetime(df.index, dayfirst=True, infer_datetime_format=True)
float_data = df.values
pd.options.display.max_columns=50
s=float_data[300001:300008].std(axis=0)
print(s)
print(float_data[300001:300008][:,1])
m=float_data[300001:300008].mean(axis=0)
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
        print(np.array(samples)[:,0][-1])
        np.savetxt('tempip.txt',np.array(samples)[:,0][-1])
        print(np.array(targets))
        yield samples, targets

lookback = 1
step = 6
delay = 1
batch_size = 128

test_gen = generator(float_data,lookback=lookback, delay=delay,min_index=300001, max_index=3000008,step=1,batch_size=1)

model=load_model("temperature_model.h5")
p=model.predict_generator(test_gen,steps=1)
np.array(p)
print(p[:,0])
np.savetxt('tempop.txt',p[:,0])
ip=np.loadtxt('tempip.txt',dtype=float)
op=np.loadtxt('tempop.txt',dtype=float)

ip[-1]=op
op=op*s+m
export=op[1]
print(op[1])
plt.plot(ip)
plt.axvline(x=19)
plt.show()

