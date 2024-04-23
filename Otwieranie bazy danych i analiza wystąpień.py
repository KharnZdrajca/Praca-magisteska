import numpy as np 
import pandas as pd 

data = pd.read_csv('baza_danych.csv')

print(data.shape)
print(data.head())
print(data.isnull().sum())

def unikalneWartosci (kolumny):
    return list(data[kolumny].unique())

for kolumny in data.select_dtypes('object').columns:
    print(f'{col}\n{unikalneWartosci(kolumny)}\n')

def kodowanie(kol):
    data[kol] = encoder.fit_transform(data[kol])
    return encoder.classes_

for kol in data.select_dtypes('object').columns:
    print(f'{kol}\n{kodowanie(kol)}\n')
    
zmienneBinarne = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Sex', 'Smoking_History']
for kol in zmienneBinarne:
    data[kol] = data[kol].astype('uint8')
    
cel = 'Heart_Disease'

import seaborn as sns
import matplotlib.pyplot as plt

def analiza(feature):
    plt.figure(figsize=(12, 8))
    color = 'Set1'
    palette_color = sns.color_palette(color)
    sns.histplot(data[feature], kde=True, bins=30, color=palette_color[random.randint(0, len(palette_color) - 1)])
    plt.xlabel(feature)
    plt.ylabel('Liczba wystąpień')
    plt.savefig(f'{feature}_wystąpienia.png')
    plt.show()

for feature in data.columns:
    if feature != cel:
        analiza(feature)
