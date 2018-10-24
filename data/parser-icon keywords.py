"""Preprocess Icon Keywords"""
import numpy as np
import pickle as pk
import random
# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# top level params to control the script
params = {
    "useNewIconCVS": False,
    "useBiGram": False, 
    "devsize": 100, 
    "testsize":100,
    "averageWeight": False 
}


# generate icon name to idx map and idx to name map
def generate_icon_idx_bi_map():
    mp_icon_idx = {}
    mp_idx_icon = {}
    idx = 0
    for icon, keywords in data.items():
        mp_icon_idx[icon] = idx
        mp_idx_icon[idx] = icon
        idx += 1
    # print(mp_icon_idx)
    fileObject = open("iconName2IndexMap.p", 'wb')
    pk.dump(mp_icon_idx, fileObject)
    fileObject.close()

    fileObject = open("iconIndex2NameMap.p", 'wb')
    pk.dump(mp_idx_icon, fileObject)
    fileObject.close()
    # print(mp_idx_icon)
    print("dumped idx map for",len(mp_icon_idx),"icons to iconName2IndexMap.p. Together with idx to name map" )


data = {} #  icon to description, keywords
icon2collection = {} # icon to collection
c = {} # collection to entry line 
mp_key2icon = {} # map of keyword to icons

trainset = list()
testset = list()
devset = list()

def find_weight(keyword):
    if not params["averageWeight"]:
        return 1
    return 1.0/len(mp_key2icon[keyword])
    
def load_icon_description_csv(isNewIcon=True, useBiGram=False, devsz=100, testsz=100):
    # parse
    if isNewIcon:
        csv_name = "IconDescriptionClean.csv"
    else:
        csv_name = "IconDescriptionClean_oldIcons.csv"
    total_paircnt = 0
    with open(csv_name, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            mediaid, collection, name = line.rstrip().split(",")[:3]
            if collection not in c:
                c[collection] = []
            c[collection] = c[collection]+[mediaid]

            if len(line.rstrip().split("\"")) != 1:
                keywords = line.rstrip().split("\"")[1]
                keywords = keywords.split(", ")
                if name not in keywords:
                    keywords = [name] + keywords
            else:
    #             print(line, keywords)
                keywords = line.rstrip().split(",")[-1]
                if name not in keywords:
                    keywords = [name] + [keywords]
#             print(mediaid, keywords, mp_key2icon)
            for keyword in keywords:
                keyword = keyword.lower()
                if keyword not in mp_key2icon:
                    mp_key2icon[keyword] = [mediaid]
                elif mediaid not in mp_key2icon[keyword]:
                    mp_key2icon[keyword] = mp_key2icon[keyword]+[mediaid]
            
            data[mediaid] = [x.lower() for x in keywords]
            icon2collection[mediaid] = collection
            total_paircnt += len(keywords)  
#     print(total_paircnt)
#     print(len(c))

    # build map
    generate_icon_idx_bi_map()
    # generate testset and devset. Each 100 entries. the rest is for trainset 
    # pass 1: add icon description to test set 
    for collecion,icons in c.items():
        icon = icons[0]
        keyword = data[icon][0]
        testset.append((icon, keyword, find_weight(keyword)))
        data[icon] = data[icon][1:]
        testsz -= 1
    # pass 2: fill in test set in a RR fashion
    while testsz != 0:
        for collecion,icons in c.items():
            if icons == []:
                continue
            icon = icons[0]
            if data[icon] == []:
                continue
            keyword = data[icon][0]
            testset.append((icon, keyword, find_weight(keyword)))
            data[icon] = data[icon][1:]
            testsz -= 1
            if testsz == 0:
                break
    # pass 3: fill in dev set, RR fashion
    while devsz != 0:
        for collecion,icons in c.items():
            if icons == []:
                continue
            icon = icons[0]
            if data[icon] == []:
                continue
            keyword = data[icon][0]
            devset.append((icon, keyword, find_weight(keyword)))
            data[icon] = data[icon][1:]
            devsz -= 1
            if devsz == 0:
                break
        
    # pass 4: fill in 
    for icon, keywords in data.items(): 
        for keyword in keywords:
            trainset.append((icon, keyword, find_weight(keyword)))
        # bigram
        if useBiGram:
            for i in range(len(keywords)-1):
                for j in range(i+1, len(keywords)):
                    if j > i:
                        trainset.append((icon,' '.join([keywords[i], keywords[j]]), find_weight(keyword[i])+find_weight(keyword[j])))
  


load_icon_description_csv(isNewIcon = params["useNewIconCVS"], useBiGram = params["useBiGram"], devsz=params["devsize"], testsz=params["testsize"])
# print(mp_key2icon)
# data = 'ConfusedPerson.svg': ['confused person', 'confusion', 'loony', 'bewildered', 'disoriented', 'embarassed', 'puzzled', 'baffled']
# c = 'Food and drinks': ['Beer.svg', 'BurgerAndDrink.svg', 'HotDog.svg', 'Tea.svg',...
# icon2collecion = 'ExclamationMark.svg': 'Signs and symbols'
# mp_key2icon = 'Open Hand with Plant': ['OpenHandWithPlant.svg']


# for each positive entry, generate a non-overlapping (keywords) & non-same-collection negative entry.
# we maintain 1:1 pos:neg ratio in the training.txt
def generateNegative(d):
    randidx = np.random.random_integers(0, len(d)-1, 150*len(d))
    j = len(d)-1 # idx in d, until 0
    for i in range(len(randidx)):
        tryidx = randidx[i]
        pos_icon = d[j][0]        
        pos_icon_keywords = set(data[pos_icon])
        try_neg_icon_keywords = set(data[d[tryidx][0]])
        # collection not the same
        if icon2collection[d[tryidx][0]] == icon2collection[pos_icon]:
#             print(d[tryidx][0], pos_icon)
            continue
        # keywords have no overlap
        if pos_icon_keywords.intersection(try_neg_icon_keywords) == set():
            d.append((pos_icon,d[tryidx][1],0))
            j-= 1
            if j == -1:
                break
generateNegative(trainset)        
# generateNegative(devset)        
# generateNegative(testset)        

            
# print(len(trainset)+ len(testset)+ len(devset))
print("generated train,dev,test (saved in training folder) size:", len(trainset), len(devset), len(testset))
np.savetxt("training/train.txt", np.array(trainset), fmt="%s")
np.savetxt("training/test.txt", np.array(testset), fmt="%s")
np.savetxt("training/dev.txt", np.array(devset), fmt="%s")



