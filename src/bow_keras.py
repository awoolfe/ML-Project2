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
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


df['comments'] = df['comments'].apply(clean_text)
df2['comments'] = df2['comments'].apply(clean_text)




train_size = int(len(df) * 0.7)
train_posts = df['comments'][:train_size]
train_tags = df['subreddits'][:train_size]

test_posts = df['comments'][train_size:]
test_tags = df['subreddits'][train_size:]

max_words = 25_000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts)  # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 100
epochs = 1

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
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
print('Test accuracy:', score[1])

