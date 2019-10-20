from tqdm import tqdm
import csv
tqdm.pandas(desc="progress-bar")
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from src.pipeline_utils import stratified_k_folds, evaluate_acc
from gensim.models.doc2vec import TaggedDocument
import gensim

import re

df = pd.read_csv("../data/reddit_spacy_train.csv")
df2 = pd.read_csv("../data/reddit_test.csv")
df.dropna(inplace=True)
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
X_train = label_sentences(df[X_COL], 'Train')
X_test = label_sentences(df2["comments"], 'Test')
y_train = label_sentences(df[Y_COL], 'Train')
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

with open('testpred.csv', mode='w') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["Id", "Category"])
    for i in y_pred:
        csv_writer.writerow(i)
