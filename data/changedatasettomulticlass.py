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
        self.embedding_method = embedding_method
        if "word2vec"in embedding_method or "word2vec+glove" in embedding_method:
            self.model = Word2Vec()
        elif "fasttext" in embedding_method:
            self.model = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
        elif "glove" in embedding_method:
            self.model = GloVe('glove/glove.42B.300d.txt.bin', loadbinary=True)
        self.__init__icon2idx()
        
    
    def __init__icon2idx(self):
        # load icon idx map first
        fileObject = open("iconName2IndexMap.p", 'rb')
        self.mp_icon2idx = pk.load(fileObject)
#         print(self.mp_icon2idx)
        self.iconNum = len(self.mp_icon2idx)
#         print(self.iconNum)
        fileObject.close()
    
    
    def genLabel(self, icons):
        res = [0]*self.iconNum
        for icon in icons:
            res[self.mp_icon2idx[icon]] = 1
        return res

    def outPut(self):
        fileObject = open("multiclass/"+self.setname+"."+self.embedding_method[0]+".p", "wb")
        pk.dump(self.mydataset, fileObject)
        fileObject.close()

    
    def proprocess(self):
        mp_keywords2icon = {}
        # pass 1: parse the input dataset
        with open("training/"+self.setname+".txt", "r") as f:
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
#                 print(icon, keywords, label)
                if keywords not in mp_keywords2icon:
                    mp_keywords2icon[keywords] = set()
                    mp_keywords2icon[keywords].add(icon)
                elif self.setname == "train":
                    mp_keywords2icon[keywords].add(icon) 
                else:
                    # for test and dev, there should be no repeat
                    raise
#         print(mp_keywords2icon)
        self.mydataset = []
        # pass 2: generate icon and phrase embedding
        for keywords, icons in mp_keywords2icon.items():
#             print(keywords, icons)
            label = self.genLabel(icons)
            phrase_embedding = self.model[keywords.split()]
            # entries for multiclass dataset
            self.mydataset.append([np.array(phrase_embedding), np.array(label), keywords, icons])
#         print(self.mydataset)
        self.outPut()
        print("processed ",self.embedding_method, self.setname, "with", len(self.mydataset), "entries")

        

# process here
M = preprocessMultiClassData("train", ["word2vec"])
M.proprocess()
M = preprocessMultiClassData("dev", ["word2vec"])
M.proprocess()
M = preprocessMultiClassData("test", ["word2vec"])
M.proprocess()


M = preprocessMultiClassData("train", ["fasttext"])
M.proprocess()
M = preprocessMultiClassData("dev", ["fasttext"])
M.proprocess()
M = preprocessMultiClassData("test", ["fasttext"])
M.proprocess()


M = preprocessMultiClassData("train", ["glove"])
M.proprocess()
M = preprocessMultiClassData("dev", ["glove"])
M.proprocess()
M = preprocessMultiClassData("test", ["glove"])
M.proprocess()
