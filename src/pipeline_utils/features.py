from typing import List, Dict
from collections import Counter, defaultdict
from itertools import chain
import numpy as np

def ngram_tokenize(tokens: List[str], n: int, token_joiner: str = " "):
    if n == 1:
        return tokens
    else:
        return [token_joiner.join(tokens[i:i + n]) for i in
                range(len(tokens) - n + 1)]


def build_vocab(ngram_tokenized_documents: List[List[str]],
                document_classes: List[str],
                min_count: int = 2
                ) -> List[str]:
    """
    # TODO: change max_counter_size to minimum occurence instead

    :param ngram_tokenized_documents:
    :param document_classes:
    :param max_counter_size:
    :return:
    """
    # create initial vocabulary
    all_documents = list(chain.from_iterable(ngram_tokenized_documents))
    vocab = [ngram for (ngram, count) in Counter(all_documents).most_common()
             if count >= min_count]
    vocab_stoi = {k:i for i,k in enumerate(vocab)}

    # count document occurrences of ngrams for different classes
    # i.e. number of documents containing an ngram
    ngram_occurences_per_class = defaultdict(lambda: np.zeros(len(vocab)))
    ngram_occurences_all_classes = np.zeros(len(vocab))
    for doc, cls in zip(ngram_tokenized_documents, document_classes):
        doc_occurences = ngram_to(doc, vocab_stoi);
        ngram_occurences_per_class[cls] += doc_occurences
        ngram_occurences_all_classes += doc_occurences

    # compute relative ngram frequencies for different classes
    # i.e. how often do ngrams occur in this class relative to other classes
    num_classes = len(ngram_occurences_per_class)
    average_occurences = ngram_occurences_all_classes / num_classes
    ngram_freq_per_class = {}
    for cls, occurences in ngram_occurences_per_class.items():
        rel_freq = (occurences - average_occurences) / occurences.sum()
        ngram_freq_per_class[cls] = rel_freq

    # find most relatively frequent ngrams across classes
    ngram_freqs = np.zeros(len(vocab))
    for rel_freq in ngram_freq_per_class.values():
        ngram_freqs = np.maximum(ngram_freqs, rel_freq)

    # return new vocab sorted by relative frequencies
    new_vocab_itos = [vocab[i] for i in (-ngram_freqs).argsort()]
    print("-"*80)
    print("These are the top 50 terms, by relative frequency: \n")
    print("\n".join(new_vocab_itos[:50]))
    print("-"*80)
    return new_vocab_itos


def ngram_tf(ngram_example: List[str],
             vocab_stoi: Dict[str, int],
             ) -> np.ndarray:
    """Calculate ngram term frequency for tokenized text.

    :param ngram_example: text split into ngrams
    :param vocab_stoi: ngram vocabulary str-to-int mapping with UNK -> 0
    :param n: ngram
    :return: ngram term frequency for each ngram in the vocabulary
    """
    ngram_frequency = np.zeros(len(vocab_stoi))
    for word, count in Counter(ngram_example).items():
        try:
            i = vocab_stoi[word]
            ngram_frequency[i] += count
        except KeyError:
            # unknown out-of-vocabulary ngram
            ngram_frequency[0] += count
    return ngram_frequency


def ngram_to(ngram_example: List[str],
             vocab_stoi: Dict[str, int],
             ) -> np.ndarray:
    """Calculate ngram term occurence for tokenized text.

    :param ngram_example: text split into ngrams
    :param vocab_stoi: ngram vocabulary str-to-int mapping with UNK -> 0
    :return: ngram term occurence for each ngram in the vocabulary
    """
    ngram_occurence = np.zeros(len(vocab_stoi))
    for word, count in Counter(ngram_example).items():
        try:
            i = vocab_stoi[word]
            ngram_occurence[i] = 1
        except KeyError:
            # unknown out-of-vocabulary ngram
            ngram_occurence[0] = 1
    return ngram_occurence


def ngram_idf(ngram_documents: List[List[str]],
              vocab_stoi: Dict[str, int],
              n: int = 1,
              laplace_smoothing: int = 1,
              ) -> np.ndarray:
    """Calculate inverse document frequency given a list of tokenized documents.

    :param ngram_documents: List of texts split into ngrams
    :param vocab_stoi: ngram vocabulary str-to-int mapping with UNK -> 0
    :param n: ngram
    :return: ngram inverse document frequency for each ngram in the vocabulary
    """
    num_documents = len(ngram_documents)
    document_frequency = np.ones(len(vocab_stoi)) * laplace_smoothing
    for ngram_example in ngram_documents:
        ngram_occurence = ngram_to(ngram_example, vocab_stoi, n)
        document_frequency += ngram_occurence

    return np.log(num_documents / document_frequency)
