from overrides import overrides
from torch.nn import functional as F
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np
import torch

@Predictor.register('subreddit_classifier')
class SubredditPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        output_dict["all_labels"] = self._model.vocab.get_index_to_token_vocabulary('labels')
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text=text)
