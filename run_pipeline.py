import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from src.model.stackingEnsemble import stackingEnsemble

from src.pipeline_utils import stratified_k_folds, evaluate_acc, evaluate, ngram_to, ngram_tokenize, build_vocab, ngram_idf, ngram_tf
from src.model.BernoulliNaiveBayes import BernoulliNaiveBayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

if __name__ == '__main__':
    K = 2
    VOCAB_SIZE = 10_000
    OOV_TOKEN = "--UNK--"
    NGRAMS = [1,2]
    Y_COL = "subreddits"
    X_COL = "preprocessed_comments"
    MODELS = [DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(), SGDClassifier(), MultinomialNB()]
    PCA_N = 1000

    print("reading data, dropping na's and creating cross-validation folds.")
    df = pd.read_csv("data/reddit_spacy_train.csv")
    df.dropna(inplace=True)
    label_stoi = {k:i for i,k in enumerate(df[Y_COL].unique())}
    train_folds, valid_folds = stratified_k_folds(df, Y_COL, K)

    for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
        X_train = train[X_COL].to_list()
        X_valid = valid[X_COL].to_list()
        Y_train = train[Y_COL].to_list()
        Y_valid = valid[Y_COL].to_list()

        print("Tokenizing texts...")
        # assumes text has already been pre-processed, tokenized,
        # then joined with whitespace, as done in data_exploration.ipynb
        X_train = [x.split() for x in X_train]
        X_valid = [x.split() for x in X_valid]

        print("Creating ngrams...")
        # creates ngrams for different values of n
        # and concatenates into single lists for building one vocabulary
        X_train = [sum([ngram_tokenize(x, n) for n in NGRAMS], [])
                   for x in X_train]
        X_valid = [sum([ngram_tokenize(x, n) for n in NGRAMS], [])
                   for x in X_valid]

        print("creating vocabulary...")
        vocab_itos = [OOV_TOKEN] + build_vocab(X_train, Y_train)[:VOCAB_SIZE]
        vocab_stoi = {k: i for i, k in enumerate(vocab_itos)}

        # print("converting ngrams to term occurence vectors...")
        # X_train = [ngram_to(x, vocab_stoi) for x in X_train]
        # X_valid = [ngram_to(x, vocab_stoi) for x in X_valid]

        print("converting ngrams to tf-idf vectors...")
        idf = ngram_idf(X_train, vocab_stoi)
        X_train = [ngram_tf(x, vocab_stoi) * idf for x in X_train]
        X_valid = [ngram_tf(x, vocab_stoi) * idf for x in X_valid]

        print("fitting PCA to training set and transforming data...")
        pca = PCA(PCA_N)
        X_trainPCA = pca.fit_transform(X_train)
        X_validPCA = pca.transform(X_valid)

        # print("converting PCA matrix to binary features...")
        # # TODO: this doesn't seem right?
        # X_train[X_train >= 0] = 1
        # X_train[X_train < 0] = 0
        # X_valid[X_valid >= 0] = 1
        # X_valid[X_valid < 0] = 0
        

        #print("Instantiating model...")
        model = LogisticRegression()
        #model2 = MultinomialNB()
        #model3 = stackingEnsemble(MODELS)

        print("Fitting model...")
        #model.fit(np.array(X_train), np.array(Y_train))
        model.fit(np.array(X_trainPCA), np.array(Y_train))
        #model3.fit(np.array(X_train), np.array(Y_train))

        print("Evaluating model...")
        #valid_pred = model.predict(np.array(X_valid))

        valid_pred2 = model.predict(np.array(X_validPCA))
        #train_pred2 = model2.predict(np.array(X_train))
        #valid_pred3 = model3.predict(np.array(X_valid))

        print("Scikit Learn:")
        print(evaluate_acc(valid_pred2, Y_valid))
        #print("OUr model")
        #print(evaluate_acc(valid_pred, Y_valid))
        #print("ensemble Model")
        #print(evaluate_acc(valid_pred3, Y_valid))
        # print(evaluate(train_pred2, Y_train, label_stoi))
        # print(evaluate(valid_pred, Y_valid, label_stoi))
        print("-"*80)





