####################################################################
# Taken from https://www.kdnuggets.com/2018/11/multi-class-text-classification-model-comparison-selection.html

######################################################################


import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from src.pipeline_utils.validation import stratified_k_folds
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

df = pd.read_csv('../data/reddit_train.csv')
df2 = pd.read_csv('../data/reddit_test.csv')

####################################################################
# Text Preprocessing                                               #
####################################################################
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    #text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(text.split())  # delete stopwors from text
    return text


df['comments'] = df['comments'].apply(clean_text)
df2['comments'] = df2['comments'].apply(clean_text)

train_folds, valid_folds = stratified_k_folds(df, 'subreddits', 5)

average_acc = 0
for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
    X_train = train['comments'].to_list()
    X_valid = valid['comments'].to_list()
    Y_train = train['subreddits'].to_list()
    Y_valid = valid['subreddits'].to_list()

    max_words = 25_000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(X_train)  # only fit on train

    x_train = tokenize.texts_to_matrix(X_train)
    x_test = tokenize.texts_to_matrix(X_valid)

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    y_train = encoder.transform(Y_train)
    y_test = encoder.transform(Y_valid)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)



    batch_size = 50
    epochs = 1

    # Build the model
    model = Sequential()
    model.add(Dense(600, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    # df2['comments'] = encoder.classes_[model.predict_classes(x_test, batch_size=batch_size, verbose=1)]
    #
    # df2.columns = ['Id', 'Category']
    #
    # df2['Category'].to_csv("../data/predictions.csv")

    score = model.evaluate(x_test, y_test,
                          batch_size=batch_size, verbose=1)
    average_acc += score[1]
    print('Test accuracy:', score[1])
    print('-'*100)
print ('Average accuracy:', average_acc/5)

