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
    if "word2vec"in embedding_method or "word2vec+glove" in embedding_method:
        w2v = Word2Vec()
    if "fasttext" in embedding_method:
        fast = FastText('fasttext/wiki-news-300d-1M.vec.bin', loadbinary=True)
    if "glove" in embedding_method or "word2vec+glove" in embedding_method:
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
            embedding = np.array([])
            if "word2vec" in embedding_method:
                embedding = np.concatenate((embedding,w2v[keywords]), axis=0)
            if "fasttext" in embedding_method:
                embedding = np.concatenate((embedding,fast[keywords]), axis=0)
            if "glove"  in embedding_method:
                embedding = np.concatenate((embedding,glove[keywords]), axis=0)
            if "word2vec+glove" in embedding_method:
                embedding = w2v[keywords]+glove[keywords]
            if "word2vec" not in embedding_method and "fasttext" not in embedding_method \
                and "glove" not in embedding_method and "word2vec+glove" not in embedding_method:
                raise
#             print(icon, keywords, label, idx, embedding)
            # phrase_embedding: iconIdx, embedding for phrase, label -- icon name, phrase idx
            phrase_embedding.append([idx, list(embedding), float(label), icon, len(phrase_embedding)])
#             break
    print(setname, ":", phrase_embedding[-1])
    print("embedding lenght=",len(phrase_embedding[-1][1]))
    
    fileObject = open("training/"+setname+"."+'-'.join(embedding_method)+".p", "wb")
    pk.dump(phrase_embedding, fileObject)
    fileObject.close()
        
# process here

# load icon idx map first
fileObject = open("iconName2IndexMap.p", 'rb')
mp_icon2idx = pk.load(fileObject)
# print(mp_icon2idx)
fileObject.close()

# preprocess each set one by one
# parse_raw_input("train", ["word2vec"])
# parse_raw_input("dev", ["word2vec"])
# parse_raw_input("test", ["word2vec"])

# parse_raw_input("train", ["fasttext"])
# parse_raw_input("dev", ["fasttext"])
# parse_raw_input("test", ["fasttext"])

# parse_raw_input("train", ["glove"])
# parse_raw_input("dev", ["glove"])
# parse_raw_input("test", ["glove"])


# parse_raw_input("train", ["word2vec","glove"])
# parse_raw_input("dev", ["word2vec","glove"])
# parse_raw_input("test", ["word2vec","glove"])

parse_raw_input("train", ["word2vec+glove"])
parse_raw_input("dev", ["word2vec+glove"])
parse_raw_input("test", ["word2vec+glove"])
print("done")
