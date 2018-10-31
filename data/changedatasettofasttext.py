"""Preprocess train/dev/test.txt - generate the corresponding set for fasttext"""
import numpy as np

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# main process func
def parse_raw_input(setname):
    mp_keyword2icon = {}
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
#             print(icon, keywords, label)
            for keyword in keywords:
                if keyword not in mp_keyword2icon:
                    mp_keyword2icon[keyword] = set()
                mp_keyword2icon[keyword].add(icon[:-4])
#     print(mp_keyword2icon)
    fw = open("fasttext/"+setname+".fasttext.txt", "w")
    for keyword in mp_keyword2icon:
        line = []
        for iconlabel in mp_keyword2icon[keyword]:
            line += ["__label__"+iconlabel]
        line += [keyword+ "\n"]
        fw.write(' '.join(line))
    fw.close()

parse_raw_input("train")
parse_raw_input("dev")
parse_raw_input("test")

        

