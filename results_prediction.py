import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy as sp
import sklearn
import random
import time
from sklearn import preprocessing,model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

import keras
keras.backend.clear_session()

data=pd.read_csv('temporada1819.csv')
data=data.iloc[:,0:22]
data=data.drop(columns=['Div','Date','HomeTeam','AwayTeam'])
data.head()
x=data.iloc[:,6:data.shape[1]]
y=data.iloc[:,2]
encoder=LabelEncoder()
encoder.fit(y)
y=encoder.transform(y)
y=np_utils.to_categorical(y)


classifier=Sequential()
classifier.add(Dense(10,activation='relu',input_dim=12))
classifier.add(Dense(10,activation='relu'))
classifier.add(Dense(10,activation='relu'))
classifier.add(Dense(10,activation='relu'))
classifier.add(Dense(3,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.summary()

y_train=y_train.reshape(-1,3)
y_test=y_test.reshape(-1,3)
classifier.fit(X_train,y_train,batch_size=8,validation_data=(X_test,y_test),epochs=100)

