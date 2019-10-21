{
  "dataset_reader": {
    "type": "subreddit",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "truncate_long_sequences": true,
      }
    },
    "tokenizer": {
      "type": "word"
    }
  },
  "train_data_path": "data/reddit_train_bert.csv",
  "validation_data_path": "data/reddit_valid_bert.csv",
  "model": {
    "type": "subreddit_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": true
      },
      "allow_unmatched_keys": true
    },
    "text_encoder": {
      "type": "bert-sentence-pooler",
    },
    "classifier_feedforward": {
      "input_dim": 768,
      "num_layers": 2,
      "hidden_dims": [768, 20],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 4,
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}