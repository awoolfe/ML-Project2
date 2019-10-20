from tqdm import tqdm
import csv
tqdm.pandas(desc="progress-bar")
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from src.pipeline_utils import stratified_k_folds, evaluate_acc
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from gensim.models.doc2vec import TaggedDocument
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import text, sequence
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
#MODELS = [LogisticRegression(), SGDClassifier(), MultinomialNB()]


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

df = pd.read_csv("../data/reddit_train.csv")
df2 = pd.read_csv("../data/reddit_test.csv")
df['comments'] = df['comments'].apply(clean_text)
df2['comments'] = df2['comments'].apply(clean_text)
train_size = int(len(df) * 0.8)
train_posts = df['comments'][:train_size]
train_tags = df['subreddits'][:train_size]

test_posts = df['comments'][train_size:]
test_tags = df['subreddits'][train_size:]

max_words = 25_000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts)

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

#df.dropna(inplace=True)
Y_COL = "subreddits"
X_COL = "preprocessed_comments"

def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

#X_train, X_test, y_train, y_test = train_test_split(df[X_COL], df[Y_COL], random_state=0, test_size=0.3)
X_train = label_sentences(train_posts, 'Train')
X_test = label_sentences(test_posts, 'Test')
y_train = label_sentences(train_tags, 'Train')
print(y_train)
all_data = X_train + X_test


model_dbow = Doc2Vec(dm=0, vector_size=500, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors


train_vectors_dbow = get_vectors(model_dbow, len(X_train), 500, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 500, 'Test')

logreg = LogisticRegression(n_jobs=1, C=1e5)
#logreg = SGDClassifier()
logreg.fit(train_vectors_dbow, y_train)
#logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow)
#print('accuracy %s' % accuracy_score(y_pred, y_test))

# with open('testpred.csv', mode='w') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(["Id", "Category"])
#     for i in y_pred:
#         csv_writer.writerow(i)
