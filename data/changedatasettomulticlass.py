"""Preprocess train/dev/test.txt - generate the corresponding set for multi class classification"""
import numpy as np
import pickle as pk
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# main process func
def parse_raw_input(setname):
    mp_keywords2icon = {}
    with open("training/"+setname+".txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.rstrip().split()
            icon = items[0]
            keywords = items[1:-1]
            label = items[-1]
            if label == "1":
                label = "True"
            else:
                label = "False"
                continue
            keywords = ' '.join(keywords)
            #             print(icon, keywords, label)
            if keywords not in mp_keywords2icon:
                mp_keywords2icon[keywords] = set()
                mp_keywords2icon[keywords].add(icon)
            elif setname == "train":
                mp_keywords2icon[keywords].add(icon) 
            else:
                # for test and dev, there should be no repeat
                raise
    print(mp_keywords2icon)
#     fw = open("multiclass/"+setname+".txt", "w")
#     for keyword in mp_keyword2icon:
#         line = []
#         for iconlabel in mp_keyword2icon[keyword]:
#             line += ["__label__"+iconlabel]
#         line += [keyword+ "\n"]
#         fw.write(' '.join(line))
#     fw.close()

# process here

# load icon idx map first
fileObject = open("iconName2IndexMap.p", 'rb')
mp_icon2idx = pk.load(fileObject)
# print(mp_icon2idx)
fileObject.close()

# parse_raw_input("train")
# parse_raw_input("dev")
parse_raw_input("test")



