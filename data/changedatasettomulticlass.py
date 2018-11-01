"""Preprocess train/dev/test.txt - generate the corresponding set for multi class classification"""
import numpy as np
import pickle as pk
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class preprocessMultiClassData:
    """class that generates phrase embedding and labels for multi class classification"""
    def __init__(self, setname, embedding_method):    
        self.setname = setname
        if "word2vec"in embedding_method or "word2vec+glove" in embedding_method:
            self.w2v = Word2Vec()
        if "fasttext" in embedding_method:
            self.fast = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
        if "glove" in embedding_method or "word2vec+glove" in embedding_method:
            self.glove = GloVe('glove/glove.42B.300d.txt.bin', loadbinary=True)
        self.__init__icon2idx()
        
    def __init__icon2idx(self):
        # load icon idx map first
        fileObject = open("iconName2IndexMap.p", 'rb')
        self.mp_icon2idx = pk.load(fileObject)
#         print(self.mp_icon2idx)
        self.iconNum = len(self.mp_icon2idx)
#         print(self.iconNum)
        fileObject.close()

# main process func
def parse_raw_input(setname, embedding_method):
    phrase_embedding = []

        
    mp_keywords2icon = {}
    with open("training/"+setname+".txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.rstrip().split()
            icon = items[0]
            keywords = items[1:-1]
            label = items[-1]
            if label == "1":
                label = "True"
            else:
                label = "False"
                continue
            keywords = ' '.join(keywords)
            #             print(icon, keywords, label)
            if keywords not in mp_keywords2icon:
                mp_keywords2icon[keywords] = set()
                mp_keywords2icon[keywords].add(icon)
            elif setname == "train":
                mp_keywords2icon[keywords].add(icon) 
            else:
                # for test and dev, there should be no repeat
                raise
#     print(mp_keywords2icon)
#     fw = open("multiclass/"+setname+".txt", "w")
#     for keyword in mp_keyword2icon:
#         line = []
#         for iconlabel in mp_keyword2icon[keyword]:
#             line += ["__label__"+iconlabel]
#         line += [keyword+ "\n"]
#         fw.write(' '.join(line))
#     fw.close()

# process here
M = preprocessMultiClassData("test", ["word2vec"])

