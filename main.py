#!/usr/bin/env python
"""Main Entry """

import pickle as pk

# External dependencies
from trainer.searcher import Searcher
from trainer.modelparam import ModelParams

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# searcher logic 
config = {
    "nn_params": [[300]],
    "max_epochs": [10000], 
    "learning_rate": [0.0001], 
    "batch_size": [1000], 
    "dropout": [0.0],
    "verbose": [True],
#     "embedding": ["word2vec"]
#     "embedding": [["word2vec","glove"]]
    "embedding": ["glove"]
#     "embedding": ["word2vec-glove"]
}

searcher = Searcher(config)
searcher.run()
