from typing import List, Dict
from collections import Counter
import numpy as np

# TODO: factor out ngram generation

def ngram_tf(tokenized_example: List[str],
             vocab_stoi: Dict[str, int],
             n: int = 1
             ) -> np.ndarray:
    """Calculate ngram term frequency for tokenized text.

    :param tokenized_example: text split into tokens (1gram)
    :param vocab_stoi: ngram vocabulary str-to-int mapping with UNK -> 0
    :param n: ngram
    :return: ngram term frequency for each ngram in the vocabulary
    """
    ngram_frequency = np.zeros(len(vocab_stoi))
    ngram_example = [" ".join(tokenized_example[i:i + n])
                     for i in range(len(tokenized_example) - n + 1)]
    for word, count in Counter(ngram_example).items():
        try:
            i = vocab_stoi[word]
            ngram_frequency[i] += count
        except KeyError:
            # unknown out-of-vocabulary ngram
            ngram_frequency[0] += count
    return ngram_frequency

def ngram_idf(tokenized_documents: List[List[str]],
              vocab_stoi: Dict[str, int],
              n: int = 1
              ) -> np.ndarray:
    """Calculate inverse document frequency given a list of tokenized documents.

    :param tokenized_documents: List of texts split into tokens (1gram)
    :param vocab_stoi: ngram vocabulary str-to-int mapping with UNK -> 0
    :param n: ngram
    :return: ngram inverse document frequency for each ngram in the vocabulary
    """
    num_documents = len(tokenized_documents)
    document_frequency = np.ones(len(vocab_stoi))
    ngram_documents = [[" ".join(tokenized_example[i:i + n])
                        for i in range(len(tokenized_example) - n + 1)]
                       for tokenized_example in tokenized_documents]
    for ngram_example in ngram_documents:
        for word, count in Counter(ngram_example).items():
            try:
                i = vocab_stoi[word]
                document_frequency[i] += 1
            except KeyError:
                # unknown out-of-vocabulary ngram
                document_frequency[0] += 1

        return np.log(num_documents/document_frequency)

