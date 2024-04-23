import numpy as np 
import pandas as pd 
import random
import os
import matplotlib.pyplot as plt
data = pd.read_csv('baza_danych.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

def kodowanie(kol):
    data[kol] = encoder.fit_transform(data[kol])
    return encoder.classes_

for kol in data.select_dtypes('object').columns:
    print(f'{kol}\n{kodowanie(kol)}\n')
    
zmienneBinarne = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Sex', 'Smoking_History']
for kol in zmienneBinarne:
    data[kol] = data[kol].astype('uint8')

cel = 'Heart_Disease'

x = data.drop(columns=[cel], axis=1)
y = data[cel]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def wykresModelu(name, parametry, X_test, ytest):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(parametry.history['accuracy'], label='Dokładność trenowania')
    plt.title('Dokładność trenowania')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(parametry.history['loss'], label='Strata trenowania')
    plt.title('Strata trenowania')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    X_test = xtest.values.reshape((xtest.shape[0], xtest.shape[1], 1))
    ewaluacja = model.evaluate(X_test, ytest)
    plt.subplot(2, 2, 3)
    plt.plot(ewaluacja[1], label='Dokładność testowania', marker='o', linestyle='None', markersize=10)
    plt.title('Dokładność testowania')
    plt.xlabel('Dokładność')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(ewaluacja[0], label='Strata testowania', marker='o', linestyle='None', markersize=10)
    plt.title('Strata testowania')
    plt.xlabel('Strata')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_Model.png')
    plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score

X_train = xtrain.values.reshape((xtrain.shape[0], xtrain.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

parametry = model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_split=0.2)

X_test = xtest.values.reshape((xtest.shape[0], xtest.shape[1], 1))

wykresModelu('LSTM', parametry, X_test, ytest)