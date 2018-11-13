"""TF-IDF model to process Erik's benchmarks"""


# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"




class TF_IDF:
    """class to generate a dict of words that should be removed"""
    def __init__(self):
        self.__load_csv()
    
    
    def __load_csv(self):
#         trainset = self.__loadErikOveson_11_05_testset("trainset_12-2017_9-1-2018_025Unk.ss.csv")
#         minWordtestset = self.__loadErikOveson_11_05_testset("testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv")
        testset = self.__loadErikOveson_11_05_testset("testset_SingleIcon_9-1_10-22-2018_025Unk.ss.csv")
        print(testset[:10])
    
    def __loadErikOveson_11_05_testset(self, filepath):
        """load files"""
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
#                         res.append([phrase[:-1], label, originalSlideCID])
                        res.append(set(phrase[:-1].split()))
                    lineID += 1
                except:
                    print(line, items)
                    raise
        return res
    
    def computeIDF(self, res):
        """compute IDF entry"""
        pass
    
    def _computeIDF_inv_freq(self, res):
        """inverse document frequency"""
        pass
    
    def computeTF(self, ph):
        """calculate tf of each word given a phrase"""
        TF = _computeTF_adjusted_w_phrase_len(ph)
        pass
        
    def _computeTF_adjusted_w_phrase_len(self, ph):
        """f_(t,d)/number of words in d"""
        pass
        
        
    
M = TF_IDF()
