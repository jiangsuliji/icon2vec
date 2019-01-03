"""generate the corresponding set for Erik's benchmark"""
import numpy as np
import pickle as pk
# import json
from collections import namedtuple, defaultdict
import re
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# top level params to control the script
params = {
    "trainsetName": "data/trainset_12-2017_9-1-2018_025Unk.ss.csv.fasttext.txt",
    "newTrainDataName": "data/newicondata_v0.txt",
    "testsetName": "data/testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv.fasttext.txt", 
    # "description_test": "data/icon_description.csv.txt",
    "description_test": "data/John test set.txt",
    "unlabeledDataName": "data/unlabel_data_all_lang.txt",
#     "embedding_method": "word2vec",
#     "embedding_method": "glove",
    "embedding_method": "fasttext",

    'doc_count_threshold': 1 #'The minimum number of documents a word or bigram should occur in to keep it in the vocabulary.'

}

Document = namedtuple('Document', \
    'icon    icon_idx  label\
     collection    collection_idx\
     phrase  phrase_vec   norm_phrase_vec\
     is_validation is_test\
    ')
#embedding_glove embedding_word2vec embedding_fasttext


setOnlyInTrain = {'toothpaste': 111, 'butterfly': 4, 'sled': 16, 'poles': 31, 'seal': 3, 'gravestone': 2, 'ringer': 1, 'pterodactyl': 2, 'tongue': 2, 'bugundermagnifyingglass': 4, 'bee': 2, 'nose': 3, 'windchime': 6, 'securitycamerasign': 6, 'mountainscene': 3, 'gymnastfloorroutine': 5, 'hummingbird': 2, 'subtitles': 6, 'browserwindow': 6, 'fireworks': 3, 'sparrow': 1, 'partymask': 1, 'fan': 3, 'eggsinbasket': 2, 'circleswitharrows': 5, 'desertscene': 1, 'dragondance': 3, 'zebra': 2, 'tyrannosaurus': 1, 'tonguefacesolidfill': 1, 'elephant': 1, 'lips': 1, 'candycane': 1}
setOnlyInTest =  {'diskjockey': 1, 'plug': 23, 'baseballhat': 4, 'arrowcircle': 5, 'electrician': 2, 'chick': 1, 'childwithballoon': 1, 'panda': 1, 'foot': 1, 'drawingcompass': 1, 'lighthousescene': 1, 'brontosaurus': 2, 'turkeycooked': 1}


class benchmarkPreprocessor:
    """class that generates phrase embedding and labels"""
    def __init__(self):    
        self.embedding_method = params["embedding_method"]
        
        with open("data/icon idx maps.p", "rb") as f:
            self.icon2idx = pk.load(f)
            self.idx2icon = pk.load(f)

        if "word2vec" == self.embedding_method:
            self.model = Word2Vec()
        elif "fasttext" == self.embedding_method:
#             self.model = FastText('data/crawl-300d-2M-subword.vec.bin', loadbinary=True)
            self.model = FastText('C:/workshop/icon2vec/data/fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
        elif "glove" == self.embedding_method:
            self.model = GloVe('data/glove.42B.300d.txt.bin', loadbinary=True)
        
        # processing
        self.loadStopList()
        vocab_freqs, doc_counts = self.gen_vocab()
        self.E, self.Var, self.embedding = self.normlize(vocab_freqs, doc_counts)
        # self.loadCSV()
        self.loadUnlabeled()

    def normlize(self, vocab_freqs, doc_counts):
        embedding = {}
        sum = 0
        E = np.zeros(300, np.float32)
        Var = 0.0

        for term, freq in vocab_freqs.items():
            embedding[term] = self.model[[term]]
            E += embedding[term]*freq
            sum += freq
        E = np.true_divide(E, sum)

        for term, freq in vocab_freqs.items():
            tmp = embedding[term]-E
            t = 0
            for i in tmp:
                t += i**2
            Var += freq*t
        Var = np.true_divide(Var, sum)
        import math
        Var = math.sqrt(Var)
        for term, ee in embedding.items():
            embedding[term] = np.true_divide(ee-E, Var)

        # print(len(embedding))
        # print(sum)
        # print(E)
        # print(Var)
        return E, Var, embedding
        
    def normalized_embedding(self, phrase):
        rtn = np.zeros(300, np.float32)
        for token in phrase: 
            if token in self.embedding:
                rtn += self.embedding[token]
            else:
                t = self.model[[token]]-self.E
                rtn += np.true_divide(t, self.Var)
        return rtn

    def loadStopList(self):
        self.stoplist = set()
        with open("data/stoplist2") as f:
            for line in f:
                self.stoplist.add(line[:-1])
        # print(self.stoplist)


    def genLabel(self, icons):
        res = [0]*len(self.icon2idx)
        for icon in icons:
            res[self.icon2idx[icon]] = 1
        return res
    
    
    def loadCSV(self):
        """main entry to load csv"""
        self.train = self.__loadErikOveson_11_05_testset(params["trainsetName"], False)
        self.train += self.__loadErikOveson_11_05_testset(params["newTrainDataName"], False)
        
        self.test = self.__loadErikOveson_11_05_testset(params["testsetName"], True)
        self.description_test =  self.__loadErikOveson_11_05_testset(params["description_test"], True)
        print("parsed train/test:", len(self.train), len(self.test))
        print("total icons:", len(self.icon2idx))
#         print(self.icon2idx)
        # print(self.train[9:10])
        self.outPut()
        
    def loadUnlabeled(self):
        self.__loadUnlabeledData(params["unlabeledDataName"], False)


    def __loadErikOveson_11_05_testset(self, filepath, is_test):
        """load """
        # smaller. close to organic
        # larger. with designertopfeedback
        res = []
        with open(filepath, 'r', encoding="utf8") as f:
            for line in f:
                if line == "": continue
                items = line.split()
                icon = []
                for i in range(len(items)):
                    if items[i][:9] == "__label__":
                        icon.append(items[i][9:])
                    else:
                        break
                phrase = [it for it in items[i:] if not it in self.stoplist]
                res.append(Document(
                      icon = tuple(icon),
                      icon_idx = tuple([self.icon2idx[eachicon] for eachicon in icon]),
                      label = self.genLabel(icon),
                      collection = None, collection_idx = None,
                      phrase = ' '.join(phrase), phrase_vec = self.model[phrase],
                      norm_phrase_vec = self.normalized_embedding(phrase), 
                      is_validation = False, is_test = is_test))
        return res
        
    def __loadUnlabeledData(self, filepath, is_test=False):
        res = []
        len_th = 400000
        cnt = 0
        with open(filepath, 'r', encoding="utf8") as f:
            for line in f:
                if line == "": continue
                items = line.split()
                phrase = [it for it in items if not it in self.stoplist]
                res.append(Document(
                      icon = None,
                      icon_idx = None,
                      label = None,
                      collection = None, collection_idx = None,
                      phrase = ' '.join(phrase), phrase_vec = self.model[phrase],
                      norm_phrase_vec = self.normalized_embedding(phrase), 
                      is_validation = False, is_test = is_test))
                if len(res) == len_th:
                  fileObject = open("tmp/train_unlabel."+self.embedding_method+".multiclass.p."+str(cnt), "wb")
                  cnt += 1
                  pk.dump(res, fileObject)
                  fileObject.close()
                  res = []

    
    
    def gen_vocab(self, trainfilePath = params["trainsetName"]):
        # vocab_freqs: dict<token, frequency count>
        # doc_counts: dict<token, document count>
        vocab_freqs = defaultdict(int)
        doc_counts = defaultdict(int)
        def fill_vocab_from_doc(phrase, vocab_freqs, doc_counts):
              doc_seen = set()
              for token in phrase:
                vocab_freqs[token] += 1
                if token not in doc_seen:
                  doc_counts[token] += 1
                  doc_seen.add(token)

        with open(trainfilePath, 'r', encoding="utf8") as f:
            for line in f:
                if line == "": continue
                items = line.split()
                for i in range(len(items)):
                    if items[i][:9] == "__label__":
                        continue
                    else:
                        break
                phrase = items[i:]
                
                fill_vocab_from_doc(phrase, vocab_freqs, doc_counts)
                
        # Filter out low-occurring terms and stopwords
        vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items() \
            if doc_counts[term] > params["doc_count_threshold"] and term not in self.stoplist)

        # Sort by frequency
        ordered_vocab_freqs = sorted(vocab_freqs.items(), key=lambda x: x[1], reverse=True)

        # filter doc counts from vocab_freqs
        filtered_doc_counts = dict((term, freq) for term, freq in doc_counts.items()\
            if term in vocab_freqs)

        # Write
        with open("tmp/vocab/vocab.txt", 'w', encoding="utf-8") as vocab_f:
            with open("tmp/vocab/vocab_freq.txt", 'w', encoding="utf-8") as freq_f:
              for word, freq in ordered_vocab_freqs:
                vocab_f.write('{}\n'.format(word))
                freq_f.write('{}\n'.format(freq))
        return vocab_freqs, filtered_doc_counts
    
    def outPut(self):
        fileObject = open("tmp/train."+self.embedding_method+".multiclass.p", "wb")
        pk.dump(self.train, fileObject)
        fileObject = open("tmp/test."+self.embedding_method+".multiclass.p", "wb")
        pk.dump(self.test, fileObject)
        fileObject = open("tmp/description_test."+self.embedding_method+".multiclass.p", "wb")
        pk.dump(self.description_test, fileObject)
        fileObject.close()

        
        # with open("tmp/train."+self.embedding_method+".multiclass.json", "w") as f:
            # json.dump(self.train, f)
        # with open("tmp/test."+self.embedding_method+".multiclass.json", "w") as f:
            # json.dump(self.test, f)




M = benchmarkPreprocessor()
