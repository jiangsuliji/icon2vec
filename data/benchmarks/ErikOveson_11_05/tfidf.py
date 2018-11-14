"""TF-IDF model to process Erik's benchmarks"""

import math

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"


class TF_IDF:
    """class to generate a dict of words that should be removed"""
    def __init__(self):
        self.__load_csv()
        self.end = None 
        self.threshold = 0.02
    
    
    def __load_csv(self):
        self.trainset, self.trainsetPrefix = self.__loadErikOveson_11_05_testset("trainset_12-2017_9-1-2018_025Unk.ss.csv")
        self.minwordset, self.minwordsetPrefix = self.__loadErikOveson_11_05_testset("testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv")
        self.testset, self.testsetPrefix = self.__loadErikOveson_11_05_testset("testset_SingleIcon_9-1_10-22-2018_025Unk.ss.csv")
#         print(self.testset[:3], self.testsetPrefix[:3])
    
    def __loadErikOveson_11_05_testset(self, filepath):
        """load files"""
        res = []
        resPrefix = []
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
#                     originalSlideCID = items[0]
#                     labels = []
#                     labelentries = items[1].split()
#                     for labelentry in labelentries:
#                         labels.append(labelentry[9:])
                    if len(items)>3:
                        phrase = ','.join(items[2:])
                    else:
                        phrase = items[2]
#                     print(originalSlideCID,labels,phrase)
                    res.append((phrase[:-1].split()))
                    resPrefix.append(','.join(items[:2])+",")
                    lineID += 1
                except:
                    print(line, items)
                    raise
        return res, resPrefix
    
    
    def computeTFIDF(self):
        # TODO: remove the end exp
        self.IDFdict_testset = self.computeIDF(self.testset[:self.end])
        self.TFIDF_testset = self.computeTF(self.testset[:self.end], self.IDFdict_testset)
#         print(self.TFIDF_testset)
        
        self.IDFdict_trainset = self.computeIDF(self.trainset[:self.end])
        self.TFIDF_trainset = self.computeTF(self.trainset[:self.end], self.IDFdict_trainset)
        
        self.IDFdict_minwordset = self.computeIDF(self.minwordset[:self.end])
        self.TFIDF_minwordset = self.computeTF(self.minwordset[:self.end], self.IDFdict_minwordset)
        
    def filterMain(self):
        """filter out low TFIDF words"""
        self.trainset[:self.end] = self._filterLowTFIDFWords(self.TFIDF_trainset, self.trainset[:self.end])
        self.minwordset[:self.end] = self._filterLowTFIDFWords(self.TFIDF_minwordset, self.minwordset[:self.end])
        self.testset[:self.end] = self._filterLowTFIDFWords(self.TFIDF_testset, self.testset[:self.end])

        
    def _filterLowTFIDFWords(self, TFIDFdict, dataset):
#         print(TFIDFdict[:10])
        filteredCNT = 0
        for idx, document in enumerate(dataset):
            newdocument = []
            for word in document:
                if TFIDFdict[idx][word] >= self.threshold:
                    newdocument.append(word)
                else:
                    filteredCNT += 1
#                     print("removed:", word)
            dataset[idx] = newdocument
#             print(document)
#             print("=>", newdocument)
        print("filtered", filteredCNT, "words")
        return dataset
        
    def dumpOut(self):
        self._dumpOut(self.testset[:self.end], self.testsetPrefix, "testset")
        self._dumpOut(self.minwordset[:self.end], self.minwordsetPrefix, "minwordset")
        self._dumpOut(self.trainset[:self.end], self.trainsetPrefix, "trainset")
    
    def _dumpOut(self, res, resPrefix, fileName):
        with open("TFIDFprocessed/"+fileName, "w", encoding="utf8") as f:
            f.write("OriginalSlideCID,Label,Train\n")
            for i in range(len(res)):
                f.write(resPrefix[i])
                f.write(" ".join(res[i])+"\n")
        
        
    def computeIDF(self, res):
        """compute IDF entry"""
        return self._computeIDF_inv_freq(res)
        
        
    def _computeIDF_inv_freq(self, res):
        """inverse document frequency"""
        IDFdict = {}
        for document in res:
            wordsInDocument = set(document)
            for word in wordsInDocument:
                if not word in IDFdict:
                    IDFdict[word] = 1
                else:
                    IDFdict[word] += 1
#         print(IDFdict)
        N = len(res)# document length
        for word, wordcnt in IDFdict.items():
            IDFdict[word] = math.log10(N/wordcnt)
        return IDFdict
        
    
    def computeTF(self, res, IDFdict):
        """calculate tf of each word given a phrase"""
        return self._computeTF_adjusted_w_phrase_len(res, IDFdict)
        
        
    def _computeTF_adjusted_w_phrase_len(self, res, IDFdict):
        """f_(t,d)/number of words in d"""
        TFIDF = []
        for idx, document in enumerate(res):
            wordsCnt = {}
            for word in document:
                if not word in wordsCnt:
                    wordsCnt[word] = 1
                else:
                    wordsCnt[word] += 1
            for word, cnt in wordsCnt.items():
                wordsCnt[word] = cnt/len(document)*IDFdict[word]
            TFIDF.append(wordsCnt)
        return TFIDF
        
    
M = TF_IDF()
M.computeTFIDF()
M.filterMain()
M.dumpOut()






