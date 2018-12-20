"""Pretrained embeddings"""
import numpy as np
import os.path
import warnings
import io
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models as gs
from gensim.scripts.glove2word2vec import glove2word2vec

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class Word2Vec:
    """word2vec model, allowing us to compute phrases"""
    def __init__(self):
        """Constructor for the Word2Vec model"""
        model_path='word2vec/GoogleNews-vectors-negative300.bin'
        if not os.path.exists(model_path):
            print(str.format('{} not found. Either provide a different path, or download binary from '
                             'https://code.google.com/archive/p/word2vec/ and unzip', model_path))
        self.model = gs.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.dimension = 300 # default 300 for googlenews pretrained model
        
    def __getitem__(self, keywords):
        """Get the vector sum of all tokens in a phrase

        Args:
            item: Phrase to be converted into a vector sum

        Returns:
            phr_sum: Bag-of-words sum of the tokens in the phrase supplied
        """
        phr_sum = np.zeros(self.dimension, np.float32)

        for key in keywords:
            if key in self.model:
                phr_sum += self.model[key]
        return phr_sum


    def __setitem__(self, key, value):
        self.model[key] = value
        
        
class FastText:
    """word2vec model, allowing us to compute phrases"""
    def __init__(self, model_path, loadbinary=True):
        if not os.path.exists(model_path):
            print(str.format('{} not found. get fasttext ', model_path))
        if not loadbinary:
            # load vec and save to bin
            self.load_and_store(model_path)
        else:
            self.model = gs.KeyedVectors.load_word2vec_format(model_path, binary = True)
            # print(self.model.most_similar('car'))
#             print(self.model['hello'])

    def __getitem__(self, keywords):
        phr_sum = np.zeros(300, np.float32)

        for key in keywords:
            if key in self.model:
                phr_sum += self.model[key]
        return phr_sum


    def __setitem__(self, key, value):
        self.model[key] = value
        
    def load_and_store(self, model_path):
        if not os.path.exists(model_path):
            print(str.format('{} not found. get fasttext ', model_path))
        self.model = gs.KeyedVectors.load_word2vec_format(model_path)
        self.model.save_word2vec_format(model_path+".bin", binary=True)
#         print(self.model['hello'])

        
class GloVe:
    def __init__(self, model_path, loadbinary=True):
        if not os.path.exists(model_path):
            print(str.format('{} not found. get GloVe ', model_path))
        if not loadbinary:
            # load vec and save to bin
            self.load_and_store(model_path)
        else:
            self.model = gs.KeyedVectors.load_word2vec_format(model_path, binary = True)
#             print(self.model.most_similar('car'))
#             print(self.model['hello'])

    def __getitem__(self, keywords):
        phr_sum = np.zeros(300, np.float32)

        for key in keywords:
            if key in self.model:
                phr_sum += self.model[key]
        return phr_sum

    def __setitem__(self, key, value):
        self.model[key] = value
        
    def load_and_store(self, model_path):
        glove2word2vec(model_path, model_path+'.bin')
        print("loaded", model_path)
        self.model = gs.KeyedVectors.load_word2vec_format(model_path+'.bin', binary=False)
        print(self.model['hello'])
        self.model.save_word2vec_format(model_path+".bin", binary=True)

        