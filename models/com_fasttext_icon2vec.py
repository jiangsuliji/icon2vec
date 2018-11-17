"""combined model"""

import pickle as pk
import numpy as np

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# fasttext_model_path = "../data/benchmarks/ErikOveson_11_05/fasttext experiment/model.bin"

    
# entry format: idx, embedding, label, icon name, phrase idx
fileObject = open("../data/iconName2IndexMap.p", 'rb')
mp_icon2idx = pk.load(fileObject)
fileObject.close()
# print(mp_icon2idx)

def __genLabelIdx(label):
    if label+".svg" in mp_icon2idx:
        return mp_icon2idx[label+".svg"]
    if label + "_LTR.svg" in mp_icon2idx:
        return mp_icon2idx[label+"_LTR.svg"]
    if label[-7:] == "Outline" and label[:-7]+"Solid.svg" in mp_icon2idx:
        return mp_icon2idx[label[:-7]+"Solid.svg"]
    if label == "Man":
        return 0
    if label == "CurveCounterclockwise":
        return mp_icon2idx["CurveClockwise.svg"]
    if label == "LineCurveCounterclockwise":
        return mp_icon2idx["LineCurveClockwise.svg"]
    if label == "BoardRoom":
        return mp_icon2idx["Boardroom.svg"]
    if label == "Australia":
        return mp_icon2idx["Australlia.svg"]
    if label == "Workflow":
        return mp_icon2idx["WorkFlow.svg"]
    if label == "HummingBird":
        return mp_icon2idx["Hummingbird.svg"]
    if label == "Ladybug":
        return mp_icon2idx["LadyBug.svg"]
    if label == "JapaneseDolls":
        return mp_icon2idx["JapaneseDoll.svg"]
    if label == "PlayingCards":
        return mp_icon2idx["PlayingCard.svg"]
    print("missing:", label)


fast = []
with open("tmp/fasttext.pred.txt", "r", encoding="utf-16") as f:
    for line in f:
        entry = []
        items = line.split()
        if len(items) == 0:
            print(len(items))
            continue
#         print(items)
        for i in range(len(items)//2):
            label = items[i*2][9:]
            label = __genLabelIdx(label)
            
            score = float(items[i*2+1])
            entry.append([label, score])
        entry.sort(key=lambda x:x[0])
        fast.append(entry)

print(len(fast), fast[0])


fileObject = open("./tmp/fasttext.p", 'wb')
pk.dump(fast, fileObject)










