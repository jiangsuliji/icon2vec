# what's inside the benchmark???
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

trainset = pk.load(open("../trainset_12-2017_9-1-2018_025Unk.ss.csv.fasttext.multiclass.p", "rb"))
testset = pk.load(open("../testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv.fasttext.p", "rb"))

iconidx2nameMap = pk.load(open("../../iconIndex2NameMap.p", "rb"))
topicons = []

print("trainset size:",len(trainset), "testset size:", len(testset))
print("total icons:", len(iconidx2nameMap))
# print(trainset[0:2])

def printHistSortedByValue(d):
    # print hist of a d
    x = [k for k,v in sorted(d.items(), key=lambda x:x[1], reverse=True) if d[k] > 100]
    y = [d[k] for k in x]
#     print(y)
#     print(d)
    plt.bar(np.arange(len(x)), y)
    plt.xticks(np.arange(len(x)), [iconidx2nameMap[x[i]] for i in range(len(x)) ])
    plt.show()

def printHistSortedByKey(d):
    x = [k for k in sorted(d.keys()) if d[k] > 20]
    plt.bar(x, [d[xx] for xx in x])
    plt.show()

def process(dataset, startstr):
    icon2Cnt = {} # find the icon distribution
    phraseLen2Cnt = {} # find the phrase length distribution
    nodupphraseLen2Cnt = {} # phrase without duplicate, len distribution
    for entry in dataset:
        if len(entry[2]) != 1:
            continue
        iconIdx = 0
        for idx, value in enumerate(entry[1]):
            if value == 1:
                iconIdx = idx
                break
        
        phraseLen = len(entry[3].split())
        nodupphraseLen = len(set(entry[3].split()))
        if iconIdx not in icon2Cnt:
            icon2Cnt[iconIdx] = 0
        if phraseLen not in phraseLen2Cnt:
            phraseLen2Cnt[phraseLen] = 0
        if nodupphraseLen not in nodupphraseLen2Cnt:
            nodupphraseLen2Cnt[nodupphraseLen] = 0
        icon2Cnt[iconIdx] += 1
        phraseLen2Cnt[phraseLen] += 1
        nodupphraseLen2Cnt[nodupphraseLen] += 1
    print(startstr, "status:")
    print("total icons", len(icon2Cnt))
#     print("phraseLen distribution:", phraseLen2Cnt)
#     print("noduplicate phraseLen distribution:", nodupphraseLen2Cnt)
    
#     printHistSortedByKey(nodupphraseLen2Cnt)
#     printHistSortedByValue(icon2Cnt)
    topicons = [(iconidx2nameMap[k], k,v) for k,v in sorted(icon2Cnt.items(), key=lambda x:x[1], reverse=True)[:20] ]
    print(topicons)
    dtopicons = {}
    i = 0
    newidxmap = {}
    for icon, iconidx, cnt in topicons:
        dtopicons[iconidx] = 0
        newidxmap[iconidx] = i
        i += 1
    print("icon cnt map", dtopicons)

    print(newidxmap)
    
#     strain = []
#     sdev = []
#     stest = []
#     addedphrase = set()
#     for entry in dataset:
#         if entry[3] in addedphrase:
#             continue
#         phraseLen = len(entry[3].split())
#         if phraseLen < 4 or phraseLen > 100:
#             continue
        
#         iconIdx = 0
#         for idx, value in enumerate(entry[1]):
#             if value == 1:
#                 iconIdx = idx
#                 break
#         if not iconIdx in dtopicons:
#             continue
#         if dtopicons[iconIdx] >= 1000:
#             continue
            
#         entry[1] = [0]*20
#         entry[1][newidxmap[iconIdx]] = 1
# #         entry[1][newidxmap]
#         addedphrase.add(entry[3])
    
#         if dtopicons[iconIdx] <800:
#             strain.append(entry)
#         elif dtopicons[iconIdx] < 900:
#             sdev.append(entry)
#         else:
#             stest.append(entry)
#         dtopicons[iconIdx] += 1
    
    
#     print(len(strain), len(stest), len(sdev))
    
#     pk.dump(strain, open("../trainset1000_train.pk", "wb"))
#     pk.dump(sdev, open("../trainset1000_dev.pk", "wb"))
#     pk.dump(stest, open("../trainset1000_test.pk", "wb"))
#     print(strain[0])
    
        
# process(trainset, "trainset")
process(testset, "testset")