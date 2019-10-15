import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

from src.pipeline_utils import stratified_k_folds, evaluate_acc, ngram_to, ngram_tokenize
from src.model.BernoulliNaiveBayes import BernoulliNaiveBayes
from sklearn.naive_bayes import BernoulliNB

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

    train_folds, valid_folds = stratified_k_folds(df, Y_COL, K)
    for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
        X_train = [ngram_to(x, vocab_stoi, NGRAM) for x in train[X_COL].to_list()]
        X_valid = [ngram_to(x, vocab_stoi, NGRAM) for x in valid[X_COL].to_list()]
        Y_train = train[Y_COL].to_list()
        Y_valid = valid[Y_COL].to_list()

        model = BernoulliNaiveBayes()
        model2 = BernoulliNB()

        model.fit(np.array(X_train), np.array(Y_train))
        model2.fit(np.array(X_train), np.array(Y_train))

        valid_pred = model.predict(np.array(X_valid))
        valid_pred2 = model2.predict(np.array(X_valid))
        print(valid_pred)
        print(np.array(Y_valid))
        print(evaluate_acc(valid_pred, Y_valid))
        print(evaluate_acc(valid_pred2, Y_valid))
        print("-"*80)





