#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.searcher import Searcher

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

config = {
    "nn_params": [[300]],
    "max_epochs": [1000], 
    "learning_rate": [0.01, 0.003, 0.001], 
    "batch_size": [1024, 2048], 
    "dropout": [0.1, 0.0],
#     "embedding": ["word2vec"]
    "embedding": ["glove"]
}

searcher = Searcher(config)
searcher.run()

