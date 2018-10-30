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
from itertools import product
import sys, os


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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Searcher:
    """Class for run train/eval using a range of parameter settings"""

    # define the model
    def __init__(self, config):
        self.config = config
        self.preload_dataset(self.config["embedding"])
        self.preload_misc()
        self.fetch_all_modelparam_list()
        
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
        
    def fetch_all_modelparam_list(self):
        self.modelParamsSettingList = [dict(zip(self.config, v)) for v in product(*self.config.values())]
        print("Total Length of Exp=", len(self.modelParamsSettingList))
        
    def create_modelParams(self, s):
        return ModelParams(in_dim=300, max_epochs=s["max_epochs"], batch_size=s["batch_size"], 
        learning_rate=s["learning_rate"], dropout=s["dropout"], class_threshold=.5, nn_params = s["nn_params"])
        
    def create_trainer(self, s):
        return Text2Vec(self.create_modelParams(s), num_icons = len(self.mp_icon2idx), trainset = self.trainset[s["embedding"]], 
                        devset = self.devset[s["embedding"]], testset = self.testset[s["embedding"]])
    
    def run_one_setting(self, s):
        trainer = self.create_trainer(s)
        rtn = trainer.train()
        trainer.close()
        del trainer
        return rtn
    
    
    def run(self):
        max_accuracy_top2 = [[0,0],[0,0]] 
        max_s = None
        for expidx, s in enumerate(self.modelParamsSettingList):
            print(expidx, "EXP setting:",s)
#             blockPrint()
            res = self.run_one_setting(s)
#             enablePrint()
            print(res)
            if res[0][1] > max_accuracy_top2[0][1]:
                max_accuracy_top2 = res
                max_s = s
        print("Finally!!!!")
        print(max_accuracy_top2, max_s)
            
            
        
        
        
        
        
        