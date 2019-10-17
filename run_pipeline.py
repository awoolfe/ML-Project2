import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from src.model.stackingEnsemble import stackingEnsemble

from src.pipeline_utils import stratified_k_folds, evaluate_acc, evaluate, ngram_to, ngram_tokenize, build_vocab
from src.model.BernoulliNaiveBayes import BernoulliNaiveBayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

if __name__ == '__main__':
    K = 5
    VOCAB_SIZE = 5
    OOV_TOKEN = "--UNK--"
    NGRAM = 1
    Y_COL = "subreddits"
    X_COL = "preprocessed_comments"

    # TODO: handle multiple ngrams at once
    df = pd.read_csv("data/reddit_spacy_train.csv")
    df.dropna(inplace=True)
    label_stoi = {k:i for i,k in enumerate(df[Y_COL].unique())}

    print("creating ngrams")
    df[X_COL] = df[X_COL].apply(lambda x: x.split())
    ngrams = [ngram_tokenize(x, NGRAM) for x in df[X_COL]]

    print("creating vocabulary")
    vocab_itos = [OOV_TOKEN] + build_vocab(ngrams, df[Y_COL].to_list(), NGRAM)[:VOCAB_SIZE-1]
    vocab_stoi = {k:i for i,k in enumerate(vocab_itos)}
    print("creating folds")
    train_folds, valid_folds = stratified_k_folds(df, Y_COL, K)

    models = [DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(), SGDClassifier(), BernoulliNaiveBayes()]

    for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
        print("Fold: ", i)
        print("Formatting data...")
        X_train = [ngram_to(x, vocab_stoi, NGRAM) for x in train[X_COL].to_list()]
        X_valid = [ngram_to(x, vocab_stoi, NGRAM) for x in valid[X_COL].to_list()]
        Y_train = train[Y_COL].to_list()
        Y_valid = valid[Y_COL].to_list()

        print("Instantiating model...")
        model = BernoulliNaiveBayes()
        model2 = BernoulliNB()
        model3 = stackingEnsemble(models)

        print("Fitting model...")
        model.fit(np.array(X_train), np.array(Y_train))
        model2.fit(np.array(X_train), np.array(Y_train))
        model3.fit(np.array(X_train), np.array(Y_train))

        print("Evaluating model...")
        valid_pred = model.predict(np.array(X_valid))

        valid_pred2 = model2.predict(np.array(X_valid))
        train_pred2 = model2.predict(np.array(X_train))
        valid_pred3 = model3.predict(np.array(X_valid))

        print("Scikit Learn:")
        print(evaluate_acc(valid_pred2, Y_valid))
        print("OUr model")
        print(evaluate_acc(valid_pred, Y_valid))
        print("ensemble Model")
        print(evaluate_acc(valid_pred3, Y_valid))
        # print(evaluate(train_pred2, Y_train, label_stoi))
        # print(evaluate(valid_pred, Y_valid, label_stoi))
        print("-"*80)





