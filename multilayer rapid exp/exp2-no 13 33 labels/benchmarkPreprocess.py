"""generate the corresponding set for Erik's benchmark"""
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
    "trainsetName": "data/trainset_12-2017_9-1-2018_025Unk.ss.csv",
    "testsetName": "data/testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv", 
    
#     "embedding_method": "word2vec",
#     "embedding_method": "glove",
    "embedding_method": "fasttext",
}

setOnlyInTrain = {'toothpaste': 111, 'butterfly': 4, 'sled': 16, 'poles': 31, 'seal': 3, 'gravestone': 2, 'ringer': 1, 'pterodactyl': 2, 'tongue': 2, 'bugundermagnifyingglass': 4, 'bee': 2, 'nose': 3, 'windchime': 6, 'securitycamerasign': 6, 'mountainscene': 3, 'gymnastfloorroutine': 5, 'hummingbird': 2, 'subtitles': 6, 'browserwindow': 6, 'fireworks': 3, 'sparrow': 1, 'partymask': 1, 'fan': 3, 'eggsinbasket': 2, 'circleswitharrows': 5, 'desertscene': 1, 'dragondance': 3, 'zebra': 2, 'tyrannosaurus': 1, 'tonguefacesolidfill': 1, 'elephant': 1, 'lips': 1, 'candycane': 1}
setOnlyInTest =  {'diskjockey': 1, 'plug': 23, 'baseballhat': 4, 'arrowcircle': 5, 'electrician': 2, 'chick': 1, 'childwithballoon': 1, 'panda': 1, 'foot': 1, 'drawingcompass': 1, 'lighthousescene': 1, 'brontosaurus': 2, 'turkeycooked': 1}


class benchmarkPreprocessor:
    """class that generates phrase embedding and labels"""
    def __init__(self):    
        self.embedding_method = params["embedding_method"]
        self.icon2idx = {}
        if "word2vec" == self.embedding_method:
            self.model = Word2Vec()
        elif "fasttext" == self.embedding_method:
#             self.model = FastText('data/crawl-300d-2M-subword.vec.bin', loadbinary=True)
            self.model = FastText('C:/workshop/icon2vec/data/fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
        elif "glove" == self.embedding_method:
            self.model = GloVe('data/glove.42B.300d.txt.bin', loadbinary=True)
            
        self.loadStopList()
        self.loadCSV()
        

    
    def loadStopList(self):
        self.stoplist = set()
        with open("data/stoplist2") as f:
            for line in f:
                self.stoplist.add(line[:-1])
#         print(self.stoplist)


    def genLabel(self, icons):
        res = [0]*len(self.icon2idx)
        for icon in icons:
            res[self.icon2idx[icon]] = 1
        return res
    
    
    def loadCSV(self):
        """main entry to load csv"""
        self.train = self.__loadErikOveson_11_05_testset(params["trainsetName"])        
        self.test = self.__loadErikOveson_11_05_testset(params["testsetName"])        
        print("parsed train/test:", len(self.train), len(self.test))
        print("total icons:", len(self.icon2idx))
#         print(self.icon2idx)
#         self.__process_method_0()
#         print(self.train[10])
    
    
    def __loadErikOveson_11_05_testset(self, filepath):
        """load """
        # smaller. close to organic
        # larger. with designertopfeedback
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
                        label = labelentry[9:].lower()
                        if len(label) >= 5 and label[-5:] == "solid":
                            label = label[:-5]
                        elif len(label) >= 7 and label[-7:] == "outline":
                            label = label[:-7]
                        elif len(label) >= 9 and label[-9:] == "solidfill":
                            label = label[:-9]
                        elif len(label) >= 16 and label[-16:] == "counterclockwise":
                            label = label[:-16] + "clockwise" 
                        
                        if label in setOnlyInTest.keys() or label in setOnlyInTrain.keys():
                            continue
                        if not label in self.icon2idx:
                            self.icon2idx[label] = len(self.icon2idx)
                        labels.append(label)
                    
                    if len(items)>3:
                        phrase = ','.join(items[2:])
                    else:
                        phrase = items[2]
                    phrase = phrase.split()
#                     print(originalSlideCID,labels,phrase)
                    if len(labels) > 0:
                        res.append([phrase, labels, originalSlideCID])
                    lineID += 1
                except:
                    print(line, items)
                    raise
        return res
    
                
    def outPut(self):
        fileObject = open("data/train."+self.embedding_method+".multiclass.remove33_13.p", "wb")
        pk.dump(self.traind, fileObject)
        fileObject = open("data/test."+self.embedding_method+".multiclass.remove33_13.p", "wb")
        pk.dump(self.testd, fileObject)
        fileObject.close()

    def proprocess(self):
        self.traind = self._proprocess(self.train)
        self.testd = self._proprocess(self.test)
        self.outPut()
        
    
    def _proprocess(self, rawdataset):
        """ generate icon and phrase embedding: embedding, iconidx, iconName, Phrase """
        d = []
        for idx, item in enumerate(rawdataset):
            phrase, labels = item[0], item[1]
#             print(idx, phrase, labels)
            phrase_embedding = self.model[phrase]
            d.append([np.array(phrase_embedding),np.array(self.genLabel(labels)), ' '.join(phrase), labels])
#             if idx == 1:
#                 break
#         print(d[0])
        print("processed ",self.embedding_method, "with", len(d), "entries;")
        return d


M = benchmarkPreprocessor()
M.proprocess()
