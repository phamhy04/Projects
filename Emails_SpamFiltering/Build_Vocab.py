
import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import words

vocab = {}

base_path = 'O:\My_Documents\MACHINE_LEARNING\Datasets'
data = pd.read_csv(os.path.join(base_path, 'emails.csv'))
nltk.download('words')
set_words = words.words()

def build_vocab(curr_mail):
    for word in curr_mail:
        if word.lower() not in vocab and word.lower() in set_words:
            vocab[word] = len(vocab)
            

if __name__ == "__main__":
    for i in range(data.shape[0]):
        curr_mail = data.iloc[i, 0].split()
        print(f"Current email is {i+1}/{data.shape[0]} and the length of vocabulary is {len(vocab)}")             
        build_vocab(curr_mail)
        
    #   Write vocab file into file.txt
    file = open(os.path.join(base_path, "vocab.txt"), 'w')
    file.write(str(vocab))
    file.close()