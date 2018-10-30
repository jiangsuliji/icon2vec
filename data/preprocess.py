"""Preprocess train/dev/test.txt - generate embedding for word2vec, fasttext, GloVe"""
import numpy as np
import pickle as pk
from pretrained_embeddings import Word2Vec
from pretrained_embeddings import FastText
from pretrained_embeddings import GloVe 
# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

# main process func
def parse_raw_input(setname, embedding_method):
    phrase_embedding = []
    if embedding_method == "word2vec":
        w2v = Word2Vec()
    elif embedding_method == "fasttext":
        fast = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
    elif embedding_method == "glove":
        glove = GloVe('glove/glove.42B.300d.txt.bin', loadbinary=True)

    with open("training/"+setname+".txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.rstrip().split()
            icon = items[0]
            keywords = items[1:-1]
            label = items[-1]
#             print(icon, keywords, label)
            idx = mp_icon2idx[icon]
            
            # find embedding for phrase
            if embedding_method == "word2vec":
                embedding = w2v[keywords]
            elif embedding_method == "fasttext":                
                embedding = fast[keywords]
            elif embedding_method == "glove":
                embedding = glove[keywords]
            else:
                raise
#             print(icon, keywords, label, idx, embedding)
            # phrase_embedding: iconIdx, embedding for phrase, label -- icon name, phrase idx
            phrase_embedding.append([idx, list(embedding), float(label), icon, len(phrase_embedding)])
#             break
    print(setname, ":", phrase_embedding[-1])
    
    fileObject = open("training/"+setname+"."+embedding_method+".p", "wb")
    pk.dump(phrase_embedding, fileObject)
    fileObject.close()
        
# process here

# load icon idx map first
fileObject = open("iconName2IndexMap.p", 'rb')
mp_icon2idx = pk.load(fileObject)
# print(mp_icon2idx)
fileObject.close()

# preprocess each set one by one
parse_raw_input("train", "word2vec")
parse_raw_input("dev", "word2vec")
parse_raw_input("test", "word2vec")

# parse_raw_input("train", "fasttext")
# parse_raw_input("dev", "fasttext")
# parse_raw_input("test", "fasttext")

parse_raw_input("train", "glove")
parse_raw_input("dev", "glove")
parse_raw_input("test", "glove")

print("done")
