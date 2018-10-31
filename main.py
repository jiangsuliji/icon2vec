#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.searcher import Searcher

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

config = {
    "nn_params": [[600]],
    "max_epochs": [1000], 
    "learning_rate": [0.003], 
    "batch_size": [1024], 
    "dropout": [0.0],
    "verbose": [True],
#     "embedding": ["word2vec"]
    "embedding": ["word2vec-glove"]
}

searcher = Searcher(config)
searcher.run()

