import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('dane.csv')

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'mieszkalny':1, 'handlowy':2}
    return word_dict[word]

X['przeznaczenie'] = X['przeznaczenie'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))