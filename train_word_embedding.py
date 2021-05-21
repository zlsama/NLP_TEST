# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:31:53 2021

@author: zhangli
"""

from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv('train_set.csv',sep='\t',nrows=10000)
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(df['label']))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


inputs = keras.Input(shape=(None,), dtype="int32")

x = layers.Embedding(len(word_index)+1, 50,input_length=MAX_SEQUENCE_LENGTH)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(32))(x)
outputs = layers.Dense(labels.shape[1], activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()


train,test,train_y,test_y = train_test_split(data,labels,test_size=0.3,random_state=666)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

model.fit(train,train_y,batch_size=32,epochs=32)

preds = model.predict(test)
pred = np.argmax(preds,axis=1)

y_true = np.argmax(test_y,axis=1)

print(f1_score(y_true, pred, average='macro'))