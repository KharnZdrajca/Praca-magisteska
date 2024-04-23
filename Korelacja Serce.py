import numpy as np 
import pandas as pd 
data = pd.read_csv('baza_danych.csv')


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

korelacja = data.corr()
korelacja[cel].sort_values(ascending=False)

data2 = data.drop(columns=[cel], axis=1)
data2.corrwith(data[cel]).plot.bar(figsize=(10, 8), title=f'Wykres korelacji', rot=90, grid=True)
plt.savefig('Wykres_korelacji.png')
plt.show()
plt.figure(figsize=(12, 10))
sns.heatmap(korelacja, fmt='.2f', annot=True)
plt.title('Macierz korelacji')
plt.savefig('Macierz_korelacji.png')
plt.show()