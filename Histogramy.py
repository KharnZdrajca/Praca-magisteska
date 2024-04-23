import numpy as np 
import pandas as pd 
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('baza_danych')

def kodowanie(kol):
    data[kol] = encoder.fit_transform(data[kol])
    return encoder.classes_

for kol in data.select_dtypes('object').columns:
    print(f'{kol}\n{kodowanie(kol)}\n')
    
zmienneBinarne = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Sex', 'Smoking_History']
for kol in zmienneBinarne:
    data[kol] = data[kol].astype('uint8')
    
    
def histogram(feature):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x=data[feature], y=data[target], hue=data[target], split=True, inner="quart", palette="viridis")
    plt.title(f'Histogram {feature} i {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.savefig(f'{feature}_Histogram.png')
    plt.show()