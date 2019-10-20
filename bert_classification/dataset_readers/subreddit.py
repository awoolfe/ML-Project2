from typing import Dict
import logging
import pandas as pd
import numpy as np
from typing import List

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("subreddit")
class SubredditReader(DatasetReader):
    """
    Dataset reader for AITA data
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        df = pd.read_csv(file_path)
        for comment, subreddit in zip(df.comments, df.subreddits):
            yield self.text_to_instance(comment, subreddit)

    @overrides
    def text_to_instance(self, text: str, label: List[float] = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)
