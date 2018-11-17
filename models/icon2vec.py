# Emoji2vec model
import pickle as pk
import numpy as np
import sys, math

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

def sigmoid(x):
    try:
        if x < -300:
            return 0.0
        return 1 / (1 + math.exp(-x))
    except:
        print("OVERFLOW!!!!!!!!!!!!!!!!!!!!",x)
        return 0.0
    
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Model_icon2vec:
    """class for calling icon2vec model"""
    def __init__(self, model_path):
        self.__open_benchmark()
        
        print("loading", model_path)
        fileObject = open(model_path, 'rb')
        self.V = pk.load(fileObject)
        fileObject = open("tmp/fasttext.p", 'rb')
        self.fast = pk.load(fileObject)
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
            score = sigmoid(np.dot(phrase, self.V[icon]))
            for n in range(N):
                if score > res[n][1]:
                    res = res[:n]+[[icon, score]]+res[n:-1]
                    break
        return res
    
    def combinedcall (self, phrase, fastres, N=2):
        r = []
        d = {}
        for it in fastres:
            d[it[0]] = it[1]
        
        for icon in range(len(self.V)):
            score = sigmoid(np.dot(phrase, self.V[icon]))
            if icon in d:
                score += d[icon]
#             else:
#                 print("miss", icon)
            
            r.append([icon, score])
            
        r.sort(key=lambda x:x[1], reverse=True)
#         print(r)
        
#         rsoft = softmax([k[1] for k in r])
#         for i in range(len(rsoft)):
#             r[i] = [r[i][0], rsoft[i]]
        
#         print(r[:10])
#         print(fastres)
        
    
        
#         return self.comb1(r[:10],fastres)
#         print(r[:2])
#         raise
        return r[:2]
        
        
    def comb1 (self, a, b):
        rtn = a + b
        rtn.sort(key=lambda x:x[1], reverse=True)
        
        raise
        return rtn[:2]
        
        
    def cal_top_n(self, dataset, str, N=2, stop = sys.maxsize):
        # quick assessment for early termination:
        results = [] # for phrase - top N icons
        for ph_idx in range(min(stop, len(dataset[1]))):
#             res = self.eval(dataset[1][ph_idx])
            res = self.combinedcall(dataset[1][ph_idx], self.fast[ph_idx], 2)
            
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