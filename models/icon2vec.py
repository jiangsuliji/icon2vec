# Emoji2vec model
import pickle as pk
import numpy as np
import sys, math

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Model_icon2vec:
    """class for calling icon2vec model"""
    def __init__(self, model_path):
        self.__open_benchmark()
        
        print("loading", model_path)
        fileObject = open(model_path, 'rb')
        self.V = pk.load(fileObject)
        fileObject.close()
    
    def __open_benchmark(self):
        fileObject = open("../data/benchmarks/testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv.fasttext.p", 'rb')
        self.benchmarkMin = pk.load(fileObject)
        fileObject.close()
        
        icon_idx, phrase_embedding, labels = [], [], []
        for item in self.benchmarkMin:
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(1)
        self.benchmarkDatasetMin = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]   
        
    def sanitytest(self):
        devres = self.cal_top_n(self.benchmarkDatasetMin, "devMin1000 ", N=2,stop=sys.maxsize)

    # main func to call for evaluation    
    def eval(self, phrase, N=2):
        res = [[-100000,-100000] for _ in range(N)]
        for icon in range(len(self.V)):
            score = np.dot(phrase, self.V[icon])
            for n in range(N):
                if score > res[n][1]:
                    res = res[:n]+[[icon, score]]+res[n:-1]
                    break
        return res
        
    def cal_top_n(self, dataset, str, N=2, stop = sys.maxsize):
        # quick assessment for early termination:
        results = [] # for phrase - top N icons
        for ph_idx in range(min(stop, len(dataset[1]))):
            res = self.eval(dataset[1][ph_idx])
            result = [r[0] for r in res]
            results.append(result)
        return self.cal_metrics(results, dataset[2], dataset[0], str, N=N)
        
        
    def cal_metrics(self, results, labels, icons, str, N=2):
        # results: top N icon indices returned by Text2Vec for each phrase
        # label for each phrase-icon pair
        # icon idx for each phrase-icon pair
#         print(results)
#         print(labels)
#         print(icons)

#         if len(results) != len(labels) or len(results) != len(icons):
#             print("error: len of inputs not equal")
#             raise
        P, T, F = [-404]*N, [0]*N, [0]*N
        for i in range(len(results)):
            if labels[i] == 1.0:
                for n in range(N):
                    if icons[i] in results[i][:n+1]:
                        T[n] += 1
                    else:
                        F[n] += 1
        for n in range(N):
            P[n] = T[n]/(T[n]+F[n])
            
        self.print_top_accuracy_TP(P, T, F, str)
        return P
    
    
    def print_top_accuracy_TP(self, P, T, F, st):
        if len(P) != len(T) or len(T) != len(F):
            raise
        s = "\t"+st + "\t"
        for i in range(len(P)):
            s += "P" +str(i+1)+"="
            s += "%3.3f," %(P[i])
        s = s[:-1] + "; "
        for i in range(len(T)):
            s += "T"+str(i+1)+"="+str(T[i]) + ",F" + str(i+1)+"="+str(F[i])+","
        s = s[:-1]
        print(s)
        
        
M = Model_icon2vec("../results/minword/lr-0.0003_ep-50000_dr-0_P1-0.11462964877813395_P2-0.18149622582980043.p")

M.sanitytest()