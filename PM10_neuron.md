# git

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

pwd

cd /Users/kozlo/Downloads


%matplotlib inline
waw = pd.read_csv("gios-pjp-data.csv", sep=";")
waw.head()


cols = ['Konstancin-Jeziorna-Wierzejewskiego - pył zawieszony PM10 1-godzinny']
waw_org[cols] = waw_org[cols].replace(',','.', regex=True).astype(float)


waw_org['Konstancin-Jeziorna-Wierzejewskiego - pył zawieszony PM10 1-godzinny'].plot()
plt.xlabel('Czas pomiaru')
plt.ylabel(cols[0])
plt.legend()
plt.show()


from pandas.plotting import lag_plot
 lag_plot(waw_org['Konstancin-Jeziorna-Wierzejewskiego - pył zawieszony PM10 1-godzinny'], lag=1)
plt.title('Konstancin-Jeziorna-Wierzejewskiego - pył zawieszony PM10 1-godzinny')


waw_org = pd.read_csv("gios-pjp-data.csv", sep=";")
waw_org=waw_org.dropna()
waw_org.tail(20)


waw_org = pd.read_csv("gios-pjp-data.csv", sep=";")
waw=waw_org.dropna()
waw = waw.drop(columns=["Zanieczyszczenie"])


waw=waw.to_numpy()
num = len(waw) 
waw = waw.reshape(num,1)
waw
stepsForward = 1
xLen = 8
xChannels = 0
print("waw.shape", waw.shape)
print("Range:", waw.shape[0] - xLen + 1 - stepsForward)


len(waw)


xTrain = np.array([waw[i:i + xLen, xChannels] for i in range(waw.shape[0] - xLen + 1 - stepsForward)])
print("xTrain.shape:", xTrain.shape)
print("xTrain:\r\n", xTrain)


yChannels = [0]
yData = waw
if (stepsForward > 1):
  yTrain = np.array([yData[i:i + stepsForward, yChannels] for i in range(xLen, yData.shape[0] + 1 - stepsForward)])
else:
  yTrain = np.array([yData[i, yChannels] for i in range(xLen, yData.shape[0] + 1 - stepsForward)])
print("yTrain.shape", yTrain.shape)
print("yTrain:\r\n", yTrain)  


valLen = 3
#Расчитываем отступ между обучающими о проверочными данными, чтобы они не смешивались
xTrainLen = xTrain.shape[0]
bias = xLen + stepsForward + 2 #Выбрасываем bias записей. Небольшой резерв на 2 записи
print("xTrainLen:", xTrainLen)
print("bias:", bias)
#Берём из конечной части xTrain проверочную выборку
xVal = xTrain[xTrainLen-valLen:]
yVal = yTrain[xTrainLen-valLen:]
print("yVal:\r\n", yVal)
print("xVal:\r\n", xVal)


#Оставшуюся часть используем под обучающую выборку
xTrain1 = xTrain[:xTrainLen-valLen-bias]
print("xTrain:\r\n", xTrain1)


#data - Numpy array
def DataNormalization(data, Channels, Normalization):
  #Выбираем тип нормализации x
  #0 - нормальное распределение
  #1 - нормирование до отрезка 0-1
  if (Normalization == 0):
    scaler = StandardScaler()
  else:
    scaler = MinMaxScaler()
  #Берём только те каналы, которые указаны в аргументе функции
  resData = waw[:,Channels]
  if (len(resData.shape) == 1): #Если размерность входного массива - одномерный вектор, 
    print("Add one dimension")
    resData = np.expand_dims(resData, axis=1) #то добавляем размерность
  #Обучаем нормировщик
  scaler.fit(resData)
  #Нормируем данные
  resData = scaler.transform(resData)
  return (resData, scaler)
def getXTrainFromTimeSeries(waw, xLen, xChannels, yChannels, stepsForward, xNormalization, yNormalization, returnFlatten, valLen, convertToDerivative):
    #Если указано превращение данных в производную
  #То вычитаем поточечно из текущей точки предыдущую
  if (convertToDerivative):
    waw = np.array([(d[1:]-d[:-1]) for d in data.T]).copy().T
  #Нормализуем x
  (waw, xScaler) = DataNormalization(waw, xChannels, xNormalization)
  #Нормализуем y
  (yData, yScaler) = DataNormalization(waw, yChannels, yNormalization)
  #Формируем xTrain
  #Раскусываем исходный ряд на куски xLen с шагом в 1
  xTrain = np.array([waw[i:i + xLen, xChannels] for i in range(waw.shape[0] - xLen + 1 - stepsForward)])
  #Формируем yTrain
  #Берём stepsForward шагов после завершения текущего x
  if (stepsForward > 1):
    yTrain = np.array([yData[i:i + stepsForward, yChannels] for i in range(xLen, yData.shape[0] + 1 - stepsForward)])
  else:
    yTrain = np.array([yData[i, yChannels] for i in range(xLen, yData.shape[0] + 1 - stepsForward)])
  #Расчитываем отступ между обучающими о проверочными данными
  #Чтобы они не смешивались
  xTrainLen = xTrain.shape[0]
  bias = xLen + stepsForward + 2
  #Берём из конечной части xTrain проверочную выборку
  xVal = xTrain[xTrainLen-valLen:]
  yVal = yTrain[xTrainLen-valLen:]
  #Оставшуюся часть используем под обучающую выборку
  xTrain = xTrain[:xTrainLen-valLen-bias]
  yTrain = yTrain[:xTrainLen-valLen-bias]
  #Если в функцию передали вернуть flatten сигнал (для Dense сети)
  #xTrain и xVal превращаем в flatten
  if (returnFlatten > 0):
    xTrain = np.array([x.flatten() for x in xTrain])
    xVal = np.array([x.flatten() for x in xVal])
  return (xTrain, yTrain), (xVal, yVal), (xScaler, yScaler)
from sklearn.preprocessing import StandardScaler
xLen = 8
stepsForward = 1
xChannels = range(waw.shape[1])
yChannels = [0]
xNormalization = 0
yNormalization = 0
valLen = 3
returnFlatten = 0 
convertToDerivative = 0
(xTrain, yTrain), (xVal, yVal), (xScaler, yScaler) = getXTrainFromTimeSeries(waw, xLen, xChannels, yChannels, stepsForward, xNormalization, yNormalization, returnFlatten, valLen, convertToDerivative)


print(xTrain.shape)
print(yTrain.shape)
print(xVal.shape)
print(yVal.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
import keras.utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation

modelC = Sequential()
modelC.add(tf.keras.layers.Conv1D(50, 5, input_shape = (xTrain.shape[1], xTrain.shape[2]), activation="linear"))

modelC.add(Dense(10, activation="linear"))
modelC.add(Dense(yTrain.shape[1], activation="linear"))
modelC.compile(optimizer='Adam', loss='mse')
history = modelC.fit(xTrain, 
                    yTrain, 
                    epochs=20, 
                    batch_size=20, 
                    verbose=1,
                    validation_data=(xVal, yVal))
plt.plot(history.history['loss'], 
         label='Average absolute error on the training set')
plt.plot(history.history['val_loss'], 
         label='Average absolute error on the test set')
plt.ylabel('Average error')
plt.legend()
plt.show()

