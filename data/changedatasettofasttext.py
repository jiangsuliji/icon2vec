"""Preprocess train/dev/test.txt - generate the corresponding set for fasttext"""
import numpy as np

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# main process func
def parse_raw_input(setname):
    fw = open("training/fast"+setname+".fasttext.txt", "w")
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
            print(icon, keywords, label)
#             fw.write(" ".join(keywords))
#             fw.write("\t")
#             fw.write(icon)
#             fw.write("\t")
#             fw.write(label)
#             fw.write("\n")
    fw.close()

# parse_raw_input("train")
# parse_raw_input("dev")
parse_raw_input("test")

        
# process here

