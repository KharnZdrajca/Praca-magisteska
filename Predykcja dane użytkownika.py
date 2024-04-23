import numpy as np 
import pandas as pd 
import random
import os
import seaborn as sns
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

def wykresModelu(name, parametry, xtest, ytest):
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
    ewaluacja = model.evaluate(xtest, ytest)
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

from tensorflow.keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(128, activation='relu', input_shape=(xtrain.shape[1], 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_split=0.2)

health = input

def predykcjaDaneUżytkownika(model, daneUżytkownika):
    daneUżytkownika = np.array(user_input).reshape(1, -1)
    predykcja = model.predict(user_input)
    return predykcja

zdrowie = input("Jak oceniasz swój ogólny stan zdrowia? (źle, średnio, dobrze, bardzo dobrze, wspaniale) ")
badania = input("Jak dawno miałeś przeprowadzane ogólne badania lekarskie? (w ostatnim roku, w ostatnich dwóch latach, w ostanich 5 latach, ponad 5 lat temu, nigdy) ")
ćwiczenia = input("Czy podczas ostatniego miesiąca wykonywałeś jakieś dodatkowe (niezwiązane z pracą) aktywności fizyczne? (tak, nie) ")
serce = input("Czy masz zdiagnozowaną chorobę serca? (tak, nie) ")
skóra = input("Czy masz zdiagnozowany nowotwór skóry? (tak, nie) ")
inny = input("Czy masz zdiagnozowany nowotwór innego rodzaju? (tak, nie) ")
depresja = input("Czy masz zdiagnozowaną depresję? (tak, nie) ")
cukrzyca = input("Czy masz zdiagnozowaną cukrzycę? (tak, nie, insulinooporność, miałam w czasie ciąży) ")
artretyzm = input("Czy masz zdiagnozowany artretyzm? (tak, nie) ")
płeć = input("Podaj swoją płeć (kobieta, mężczyzna): ")
wiek = input("Podaj swój wiek w jednym z przedziałów (18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+) ")
wzrost = float(input("Podaj swój wzrost (w cm): "))
waga = float(input("Podaj swoją wagę (w kg): "))
BMI = float(input("Podaj swój wskaźnik BMI: "))
papierosy = input("Czy palisz lub paliłeś kiedyś papierosy? (tak, nie) ")
alkohol = float(input("Ile razy w ostatnim miesiący spożywałeś alkohol? "))
owoce = float(input("Ile razy w ostatnim miesiący spożywałeś owoce? "))
warzywa = float(input("Ile razy w ostatnim miesiący spożywałeś warzywa? "))
ziemniaki = float(input("Ile razy w ostatnim miesiący spożywałeś smażone ziemniaki? "))

daneUżytkownika = [zdrowie,badania,ćwiczenia,serce,skóra,inny,depresja,cukrzyca,artretyzm,płeć,wiek,wzrost,waga,BMI,papierosy,alkohol,owoce,warzywa,ziemniaki]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

def przekształcanieDanychUżytkownika(column_name, daneUżytkownika):
    temp_df = pd.DataFrame({nazwaKolumny: daneUżytkownika})
    
    przekształconaKolumna = encoder.fit_transform(temp_df[nazwaKolumny])
    
    return przekształconaKolumna

przekształconeDaneUżytkownika = []
for i, feature in enumerate(daneUżytkownika):
    if feature in ['zdrowie', 'badania', 'ćwiczenia', 'wzrost']:
        przekształconaWartość = przekształcanieDanychUżytkownika(feature, [daneUżytkownika[i]])[0]
        przekształconeDaneUżytkownika.append(przekształconaWartość)
    elif feature == 'płeć':
        przekształconaWartość = 1 if daneUżytkownika[i] == 'mężczyzna' else 0
        przekształconeDaneUżytkownika.append(przekształconaWartość)
    elif feature in ['skóra', 'inny', 'depresja', 'cukrzyca', 'artretyzm', 'papierosy']:
        przekształconaWartość = 1 if daneUżytkownika[i] == 'tak' else 0
        przekształconeDaneUżytkownika.append(przekształconaWartość)
    else:
        przekształconeDaneUżytkownika.append(daneUżytkownika[i])

przekształconeDaneUżytkownika = np.array(przekształconeDaneUżytkownika)

przekształconeDaneUżytkownika = przekształconeDaneUżytkownika.astype(float)

przekształconeDaneUżytkownika = np.array(przekształconeDaneUżytkownika).reshape(1, -1)

predykcja = predykcjaDaneUżytkownika(model, przekształconeDaneUżytkownika)
print("Predykcja dla danych użytkownika:", predykcja)