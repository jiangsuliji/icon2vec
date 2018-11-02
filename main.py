#!/usr/bin/env python
"""Main Entry """

import pickle as pk

# External dependencies
from trainer.searcher import Searcher
from trainer.model import Text2VecMulti
from trainer.modelparam import ModelParams

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
modelParams = ModelParams(in_dim=300, max_epochs=1000, batch_size=128, learning_rate=0.001, class_threshold=0.5, dropout=0.1, nn_params = [])

# embedding = "word2vec"
# embedding = "fasttext"
embedding = "glove"

fileObject = open("data/multiclass/train."+embedding+".p", 'rb')
trainset = pk.load(fileObject)
fileObject = open("data/multiclass/dev."+embedding+".p", 'rb')
devset = pk.load(fileObject)
# print(len(devset), devset[0])
fileObject = open("data/multiclass/test."+embedding+".p", 'rb')
testset = pk.load(fileObject)
fileObject.close()
print(len(trainset), len(devset), len(testset))
# print(trainset[0])

M = Text2VecMulti( model_params = modelParams, trainset = trainset, devset = devset, testset = testset)
M.train()