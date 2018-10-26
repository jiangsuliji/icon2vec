#!/usr/bin/env python
"""
param searcher to find the best P@1 P@2
"""

# External dependencies
# import tensorflow as tf
import numpy as np
from modelparam import ModelParams
from model import Text2Vec

# from os import environ
# from random import shuffle
# import sklearn.metrics as metrics
# from warnings import filterwarnings 
# filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim.models as gs
# import pickle as pk

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# top level params to control the script
params = {
    "nn_params": [[200],[300],[400],[300,300]],
    "max_epochs": [100], 
    "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001], 
    "batch_size": [4,16,32], 
    "embedding": ["word2vec", "fasttext", "glove"]
}

# nn_params =  [300] # for additional hidden layers
# modelParams = ModelParams(in_dim=300, max_epochs=300, batch_size=32, learning_rate=0.003, dropout=0.1, class_threshold=.5, nn_params = nn_params)

class Searcher:
    """Class for run train/eval using a range of parameter settings"""

    # define the model
    def __init__(self):
        self.preload_embeddings()
        
        
    def preload_embeddings(self):
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