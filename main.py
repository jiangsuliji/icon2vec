#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.searcher import Searcher

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# top level params to control the script
# config = {
#     "nn_params": [[300],[400],[300,300]],
#     "max_epochs": [1000], 
#     "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001], 
#     "batch_size": [32], 
#     "dropout": [0.0, 0.2],
#     "embedding": ["word2vec", "fasttext", "glove"]
# }

config = {
    "nn_params": [[300]],
    "max_epochs": [1000], 
    "learning_rate": [0.0001], 
    "batch_size": [32], 
    "dropout": [0.0,0.1,0.2],
#     "embedding": ["word2vec"]
    "embedding": ["word2vec", "glove"]
}

searcher = Searcher(config)
searcher.run()

