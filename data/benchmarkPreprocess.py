"""Preprocess train/dev/test.txt - generate the corresponding set for Erik's benchmark"""
import numpy as np
import pickle as pk
import re
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# top level params to control the script
params = {
#     "datasetName": "trainset_12-2017_9-1-2018_025Unk.ss.csv",
#     "datasetName": "testset_SingleIcon_9-1_10-22-2018_025Unk.ss.csv",
    "datasetName": "testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv", 
#     "embedding_method": "word2vec"
    "embedding_method": "glove"
#     "embedding_method": "fasttext"
}



class benchmarkPreprocessor:
    """class that generates phrase embedding and labels"""
    def __init__(self):    
        self.embedding_method = params["embedding_method"]
        
        if "word2vec" == self.embedding_method:
            self.model = Word2Vec()
        elif "fasttext" == self.embedding_method:
            self.model = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
        elif "glove" == self.embedding_method:
            self.model = GloVe('glove/glove.42B.300d.txt.bin', loadbinary=True)
            
        self.loadStopList()
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
   
    def loadStopList(self):
        self.stoplist = set()
        with open("stoplists/stoplist2") as f:
            for line in f:
                self.stoplist.add(line[:-1])
#         print(self.stoplist)

    def __genLabelIdx(self, label):
        if label+".svg" in self.mp_icon2idx:
            return self.mp_icon2idx[label+".svg"]
        if label + "_LTR.svg" in self.mp_icon2idx:
            return self.mp_icon2idx[label+"_LTR.svg"]
        if label[-7:] == "Outline" and label[:-7]+"Solid.svg" in self.mp_icon2idx:
            return self.mp_icon2idx[label[:-7]+"Solid.svg"]
        if label == "Man":
            return 0
        if label == "CurveCounterclockwise":
            return self.mp_icon2idx["CurveClockwise.svg"]
        if label == "LineCurveCounterclockwise":
            return self.mp_icon2idx["LineCurveClockwise.svg"]
        if label == "BoardRoom":
            return self.mp_icon2idx["Boardroom.svg"]
        if label == "Australia":
            return self.mp_icon2idx["Australlia.svg"]
        if label == "Workflow":
            return self.mp_icon2idx["WorkFlow.svg"]
        if label == "HummingBird":
            return self.mp_icon2idx["Hummingbird.svg"]
        if label == "Ladybug":
            return self.mp_icon2idx["LadyBug.svg"]
        if label == "JapaneseDolls":
            return self.mp_icon2idx["JapaneseDoll.svg"]
        if label == "PlayingCards":
            return self.mp_icon2idx["PlayingCard.svg"]
        print("missing:", label)
    
    def loadCSV(self):
        """main entry to load csv"""
        self.benchmark = self.__loadErikOveson_11_05_testset()        
        self.__process_method_0()
#         print(self.benchmark[10])
    
    def __loadErikOveson_11_05_testset(self):
        """load """
        # smaller. close to organic
        filepath = "benchmarks/ErikOveson_11_05/" + params["datasetName"]
        # larger. with designer feedback
        res = []
        with open(filepath, 'r', encoding="utf8") as f:
            lineID = 0
            for line in f:
                try:
                    if line == "\n":
                        continue
                    if lineID == 0:
                        lineID += 1
                        continue
    #                 print(line)
                    items = line.split(',')
                    originalSlideCID = items[0]
                    labels = []
                    labelentries = items[1].split()
                    for labelentry in labelentries:
                        labels.append(labelentry[9:])
                    if len(items)>3:
                        phrase = ','.join(items[2:])
                    else:
                        phrase = items[2]
    #                 print(originalSlideCID,labels,phrase)
                    for label in labels:
                        res.append([phrase, label, originalSlideCID])
                    lineID += 1
                except:
                    print(line, items)
                    raise
        return res
    
    def __process_method_0(self):
        """remove #UNK_WORD #UNK_NUMBER non character symbol"""
        for idx, item in enumerate(self.benchmark):
            phrase, label, originalSlideCID = item[0], item[1], item[2]
            items = phrase.split()
            cleaned_items = set()
            for i in items:
                if i in self.stoplist or len(i) <= 1:
                    continue
                if re.match("^[a-z_-]*$", i):
#                 if re.match("^[A-Za-z0-9_-]*$", i):
                    cleaned_items.add(i)
            self.benchmark[idx] = [list(cleaned_items), self.__genLabelIdx(label), label, originalSlideCID]
        
    
    def outPut(self):
        fileObject = open("benchmarks/"+params["datasetName"]+"."+self.embedding_method+".p", "wb")
        pk.dump(self.mydataset, fileObject)
        fileObject.close()

    
    def proprocess(self):
        """ generate icon and phrase embedding: embedding, iconidx, iconName, Phrase """
        self.mydataset = []
        for idx, item in enumerate(self.benchmark):
            phrase, labelidx = item[0], item[1]
            phrase_embedding = self.model[phrase]
            self.mydataset.append([np.array(phrase_embedding),labelidx, item[2], ' '.join(phrase)])
#             if idx == 3:
#                 break
#         prist(self.mydataset)
        self.outPut()
        print(self.mydataset[0])
        print("processed ",self.embedding_method, "with", len(self.mydataset), "entries")


M = benchmarkPreprocessor()
M.proprocess()
