#!/usr/bin/env python
"""
param searcher to find the best P@1 P@2
"""

# External dependencies
# import tensorflow as tf
import numpy as np
from trainer.modelparam import ModelParams
from trainer.model import Text2Vec

# from os import environ
# from random import shuffle
# import sklearn.metrics as metrics
# from warnings import filterwarnings 
# filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim.models as gs
import pickle as pk

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# # top level params to control the script
# config = {
#     "nn_params": [[200],[300],[400],[300,300]],
#     "max_epochs": [100], 
#     "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001], 
#     "batch_size": [4,16,32], 
#     "embedding": ["word2vec", "fasttext", "glove"]
# }

# nn_params =  [300] # for additional hidden layers
# modelParams = ModelParams(in_dim=300, max_epochs=300, batch_size=32, learning_rate=0.003, dropout=0.1, class_threshold=.5, nn_params = nn_params)

class Searcher:
    """Class for run train/eval using a range of parameter settings"""

    # define the model
    def __init__(self, config):
        self.config = config
        self.preload_dataset(self.config["embedding"])
        self.preload_misc()
        
    def preload_dataset(self, embeddingList):
        self.trainset = {}
        self.devset = {}
        self.testset = {}
        for embedding in embeddingList:
            fileObject = open("data/training/train."+embedding+".p", 'rb')
            self.trainset[embedding] = pk.load(fileObject)
            fileObject = open("data/training/dev."+embedding+".p", 'rb')
            self.devset[embedding] = pk.load(fileObject)
            # print(len(devset), devset[0])
            fileObject = open("data/training/test."+embedding+".p", 'rb')
            self.testset[embedding] = pk.load(fileObject)
            fileObject.close()
            
    def preload_misc(self):
        # load data, map
        # entry format: idx, embedding, label, icon name, phrase idx
        fileObject = open("data/iconName2IndexMap.p", 'rb')
        self.mp_icon2idx = pk.load(fileObject)
        # print(mp_icon2idx)