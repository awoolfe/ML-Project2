import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

from src.pipeline_utils import stratified_k_folds, evaluate_acc, ngram_to, ngram_tokenize
from src.model.BernoulliNaiveBayes import BernoulliNaiveBayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    K = 5
    VOCAB_SIZE = 1000
    OOV_TOKEN = "--UNK--"
    NGRAM = 1
    Y_COL = "subreddits"
    X_COL = "preprocessed_comments"

    # TODO: handle multiple ngrams at once

    df = pd.read_csv("data/reddit_spacy_train.csv")
    df.dropna(inplace=True)

    ngrams = [ngram_tokenize(x.split(), NGRAM) for x in df[X_COL]]
    vocab = Counter(list(chain.from_iterable(ngrams)))
    vocab_itos = [OOV_TOKEN] + [word for word, count in vocab.most_common(VOCAB_SIZE - 1)]
    vocab_stoi = {k:i for i,k in enumerate(vocab_itos)}

    '''TESTING SKLEARN PIPELINE AND SVM'''
    sgd = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(max_iter=10))])

    train_folds, valid_folds = stratified_k_folds(df, Y_COL, K)
    for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
        #sklearn implementation linear support vector machines
        X_train1, X_test1, y_train1, y_test1 = train_test_split(train[X_COL].to_list(), train[Y_COL].to_list(), test_size=0.3, random_state=42)

        X_train = [ngram_to(x, vocab_stoi, NGRAM) for x in train[X_COL].to_list()]
        X_valid = [ngram_to(x, vocab_stoi, NGRAM) for x in valid[X_COL].to_list()]
        Y_train = train[Y_COL].to_list()
        Y_valid = valid[Y_COL].to_list()

        model = BernoulliNaiveBayes()
        #model2 = BernoulliNB()
        dtree = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=3)
        kneighbor = KNeighborsClassifier(n_neighbors=5)

        sgd.fit(X_train1, y_train1)

        #model.fit(np.array(X_train), np.array(Y_train))
        #model2.fit(np.array(X_train), np.array(Y_train))
        #dtree.fit(np.array(X_train), np.array(Y_train))
        #kneighbor.fit(np.array(X_train), np.array(Y_train))

        #valid_pred = model.predict(np.array(X_valid))
        #valid_pred2 = model2.predict(np.array(X_valid))
        #dtree_pred = dtree.predict(np.array(X_valid))
        #kneighbor_pred = kneighbor.predict(np.array(X_valid))
        sgd_pred = sgd.predict(X_test1)
        #print(valid_pred)
        print(np.array(Y_valid))
        #print(evaluate_acc(valid_pred, Y_valid))
        #print(evaluate_acc(valid_pred2, Y_valid))
        #print(evaluate_acc(kneighbor_pred, Y_valid))
        #print(evaluate_acc(dtree_pred, Y_valid))
        print(evaluate_acc(sgd_pred, y_test1))
        print("-"*80)





