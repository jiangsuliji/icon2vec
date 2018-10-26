#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.searcher import Searcher

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# create model params
# nn_params =  [300] # for additional hidden layers
# modelParams = ModelParams(in_dim=300, max_epochs=300, batch_size=32, learning_rate=0.003, dropout=0.1, class_threshold=.5, nn_params = nn_params)



# # embedding = "word2vec"
# # embedding = "fasttext"
# embedding = "glove"

# fileObject = open("data/training/train."+embedding+".p", 'rb')
# trainset = pk.load(fileObject)
# fileObject = open("data/training/dev."+embedding+".p", 'rb')
# devset = pk.load(fileObject)
# # print(len(devset), devset[0])
# fileObject = open("data/training/test."+embedding+".p", 'rb')
# testset = pk.load(fileObject)
# fileObject.close()
# # print(len(trainset), len(devset), trainset[0])



# # create model
# model = Text2Vec(modelParams, num_icons = len(mp_icon2idx), trainset = trainset, devset = devset, testset = testset)
# model.train()


# top level params to control the script
config = {
    "nn_params": [[200],[300],[400],[300,300]],
    "max_epochs": [100], 
    "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001], 
    "batch_size": [4,16,32], 
    "embedding": ["word2vec", "fasttext", "glove"]
}

searcher = Searcher(config)


