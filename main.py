#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.searcher import Searcher
from trainer.model import Text2VecMulti

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# # searcher logic for the non-multi-class embedding
# config = {
#     "nn_params": [[600]],
#     "max_epochs": [1000], 
#     "learning_rate": [0.003], 
#     "batch_size": [1024], 
#     "dropout": [0.0],
#     "verbose": [True],
# #     "embedding": ["word2vec"]
# #     "embedding": [["word2vec","glove"]]
# #     "embedding": ["glove"]
#     "embedding": ["word2vec-glove"]
# }

# searcher = Searcher(config)
# searcher.run()

# multi-class embedding
modelParams = ModelParams(in_dim=300, max_epochs=1000, batch_size=32, learning_rate=0.003, class_threshold=0.5, dropout=0.1, nn_params = [])

M = Text2VecMulti(modelParams = modelParams, trainset = trainset, devset = devset, testset = testset)
M.train()