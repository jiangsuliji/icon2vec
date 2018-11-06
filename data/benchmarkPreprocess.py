"""Preprocess train/dev/test.txt - generate the corresponding set for Erik's benchmark"""
import numpy as np
import pickle as pk
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class benchmarkPreprocessor:
    """class that generates phrase embedding and labels"""
    def __init__(self, setname, embedding_method):    
        self.setname = setname
        self.embedding_method = embedding_method
        
#         if "word2vec"in embedding_method:
#             self.model = Word2Vec()
#         elif "fasttext" in embedding_method:
#             self.model = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
#         elif "glove" in embedding_method:
#             self.model = GloVe('glove/glove.42B.300d.txt.bin', loadbinary=True)
            
        self.__init__icon2idx()
        self.loadCSV()
        
        
    def __init__icon2idx(self):
        # load icon idx map first
        fileObject = open("iconName2IndexMap.p", 'rb')
        self.mp_icon2idx = pk.load(fileObject)
#         print(self.mp_icon2idx)
        self.iconNum = len(self.mp_icon2idx)
#         print(self.iconNum)
        fileObject.close()
    
    
    def loadCSV(self):
        self.benchmark = self.__loadErikOveson_11_05_testset()
        print(self.benchmark[0])
    
    def __loadErikOveson_11_05_testset(self):
        # smaller. close to organic
        filepath = "benchmarks/ErikOveson_11_05/testset_SingleIcon_9-1_10-22-2018_025Unk.ss.csv" 
        # larger. with designer feedback
#         filepath = "benchmarks/ErikOveson_11_05/testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss" 
        res = []
        with open(filepath, 'r', encoding="utf8") as f:
            lineID = 0
            for line in f:
                if lineID == 0:
                    lineID += 1
                    continue
                print(line)
                items = line.split(',')
                originalSlideCID = items[0]
                label = items[1][9:]
                phrase = ','.join(items[2])
#                 print(originalSlideCID,label,phrase)
                res.append([phrase, label, originalSlideCID])
                lineID += 1
        return res
    
    
    def genLabel(self, icons):
        res = [0]*self.iconNum
        for icon in icons:
            res[self.mp_icon2idx[icon]] = 1
        return res

    
    def outPut(self):
        fileObject = open("benchmarks/"+self.setname+"."+self.embedding_method[0]+".p", "wb")
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
# M = benchmarkPreprocessor("train", ["word2vec"])
# M.proprocess()
# M = benchmarkPreprocessor("dev", ["word2vec"])
# M.proprocess()
M = benchmarkPreprocessor("test", ["word2vec"])
# M.proprocess()


# M = benchmarkPreprocessor("train", ["fasttext"])
# M.proprocess()
# M = benchmarkPreprocessor("dev", ["fasttext"])
# M.proprocess()
# M = benchmarkPreprocessor("test", ["fasttext"])
# M.proprocess()


# M = benchmarkPreprocessor("train", ["glove"])
# M.proprocess()
# M = benchmarkPreprocessor("dev", ["glove"])
# M.proprocess()
# M = benchmarkPreprocessor("test", ["glove"])
# M.proprocess()