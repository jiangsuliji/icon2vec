#!/usr/bin/env python
"""Main Entry """

# External dependencies
from trainer.modelparam import ModelParams
from trainer.model import Text2Vec
import numpy as np
import pickle as pk

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# create model params
nn_params =  [300] # for additional hidden layers
modelParams = ModelParams(in_dim=300, max_epochs=1000, batch_size=32, learning_rate=0.01, dropout=0., class_threshold=.5, nn_params = nn_params)

# precessing
# load data, map
# entry format: idx, embedding, label, icon name, phrase idx
fileObject = open("data/iconName2IndexMap.p", 'rb')
mp_icon2idx = pk.load(fileObject)
# print(mp_icon2idx)

# embedding = "word2vec"
# embedding = "fasttext"
embedding = "glove"

fileObject = open("data/training/train."+embedding+".p", 'rb')
trainset = pk.load(fileObject)
fileObject = open("data/training/dev."+embedding+".p", 'rb')
devset = pk.load(fileObject)
# print(len(devset), devset[0])
fileObject = open("data/training/test."+embedding+".p", 'rb')
testset = pk.load(fileObject)
fileObject.close()
# print(len(trainset), len(devset), trainset[0])


# create model
model = Text2Vec(modelParams, num_icons = len(mp_icon2idx), trainset = trainset, devset = devset, testset = testset)
model.train()