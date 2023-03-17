import numpy as np
import pandas as pd
import os
import ast

base_path = 'O:\My_Documents\MACHINE_LEARNING\Datasets'
data = pd.read_csv(os.path.join(base_path, 'emails.csv'))
file = open(os.path.join(base_path, 'vocab.txt'), 'r')
vocab = ast.literal_eval(file.read())

X = np.zeros((data.shape[0], len(vocab)))
Y = np.zeros((data.shape[0], 1))

for i in range(data.shape[0]):
    curr_email = data.iloc[i, 0].split()
    for word in curr_email:
        if word.lower() in vocab:
            X[i, vocab[word]] += 1
            Y[i] = data.iloc[i, 1]
            
np.save(os.path.join(base_path, 'X.npy'), X)
np.save(os.path.join(base_path, 'Y.npy'), Y)

