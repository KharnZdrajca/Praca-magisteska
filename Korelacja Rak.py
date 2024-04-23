import numpy as np 
import pandas as pd 
import random
import os
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
    
print(data.info())

cel = 'Skin_Cancer'

import seaborn as sns
import matplotlib.pyplot as plt

korelacja = data.corr()
korelacja[cel].sort_values(ascending=False)
print(korelacja[cel])

wykresKor = data.drop(columns=[cel], axis=1)
wykresKor.corrwith(data[cel]).plot.bar(figsize=(10, 8), title=f'Wykres korelacji', rot=90, grid=True)
plt.savefig('Wykres_korelacji.png')
plt.show()
plt.figure(figsize=(12, 10))
sns.heatmap(korelacja, fmt='.2f', annot=True)
plt.title('Macierz korelacji')
plt.savefig('Macierz_korelacji.png')
plt.show()
