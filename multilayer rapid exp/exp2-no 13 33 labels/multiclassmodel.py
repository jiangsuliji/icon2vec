#!/usr/bin/env python
"""
File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
"""

# External dependencies
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import sys
from os import environ
from random import shuffle
import sklearn.metrics as metrics
from warnings import filterwarnings 
filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models as gs
import pickle as pk
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable all debugging logs
tf.logging.set_verbosity(tf.logging.FATAL)

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

#-----------------------------------------------------------------------
#                  multi class classifier
#-----------------------------------------------------------------------
class Text2VecMulti:
    """class for multi class model"""
    def __init__(self):
        self.num_icons = 753
        self.batch_size = 650 
        self.dropout = 0
        self.max_epochs = 3000000
        self.learning_rate = 0.0001
        self.nn_params = [300, 1500, 1500, 1500, 1500, 900]
        self.in_dim = 300
        
        self.initializeDatasetWithBenchmarkTraining()
        self.initializeModel()
        print("parsed train/test:", len(self.trainset[0]), len(self.testset[0]))
    
    
    def initializeDatasetWithBenchmarkTraining(self):
        """initialize benchmark """ 
        fileObject = open("data/train.fasttext.multiclass.remove33_13.p", 'rb')
        benchmarktrainraw = pk.load(fileObject)
        # parse the positive
        icon_idx, phrase_embedding, labels = [], [], []
        ph_idx = []
        
        # filtering dominantion classes
        iconCntDict = {}
        
        for item in benchmarktrainraw:
#             phraseLen = len(item[3].split())
#             if phraseLen > 100 or phraseLen < 4:
#                 continue
#             if len(item[2]) == 1:
#                 if not item[2][0] in iconCntDict:
#                     iconCntDict[item[2][0]] = 1
#                 elif iconCntDict[item[2][0]] > 5000:
#                     continue
#                 else:
#                     iconCntDict[item[2][0]] += 1
            
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(item[3])

#             if len(labels) == 10000:
#                 break
#             print(icon_idx, phrase_embedding, labels)
        self.trainset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
         
        fileObject = open("data/test.fasttext.multiclass.remove33_13.p", 'rb')
        benchmarkMin = pk.load(fileObject)
        fileObject.close()
       
        icon_idx, phrase_embedding, labels = [], [], []
        for item in benchmarkMin:
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(item[3])
        self.testset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
    
        
    def initializeSession(self):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)
    
    
    def initializeModel(self):
        # row - phrase input to the graph
        self.phrase_vec = tf.placeholder(tf.float32, shape=[None, self.in_dim], name='phrase_vec')
            
        # corect label
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_icons], name='y')
        
        self.trainFlag = tf.placeholder(tf.bool)
        
        # single layer multiclassifier
        if self.nn_params == []:
            W = tf.get_variable("W", shape=[self.model_params.in_dim, self.num_icons], initializer=tf.contrib.layers.xavier_initializer())
            self.logits = tf.matmul(self.phrase_vec, tf.nn.dropout(W,1-self.model_params.dropout))
            self.logits = tf.nn.relu(self.logits)
        else:
            # todo multi layer
            Wmat = []
            Bmat = []
            self.regularizer = 0
            prev = self.in_dim
            score = self.phrase_vec
            for idx, num in enumerate(self.nn_params):
                W = tf.get_variable("W"+str(idx), shape=[prev, num], initializer=tf.contrib.layers.xavier_initializer())
#                 W = tf.get_variable("W"+str(idx), shape=[prev, num], initializer=tf.truncated_normal_initializer(stddev=5e-2))
                B = tf.get_variable("B"+str(idx), shape=[num], initializer=tf.constant_initializer(0.1))
#                 W_fc1 = tf.truncated_normal([prev, num], mean=0.5, stddev=0.707)
#                 W = tf.Variable(W_fc1, name='W_fc'+str(idx))

#                 b_fc1 = tf.truncated_normal([num], mean=0.5, stddev=0.707)
#                 B = tf.Variable(b_fc1, name='b_fc'+str(idx))
                score = tf.nn.relu(tf.add(tf.matmul(score, W), B))
#                 self.regularizer += tf.nn.l2_loss(W)

                prev = num
                Wmat.append(W)
                Bmat.append(B)
            
            # add last layer
#             W = tf.get_variable("W", shape=[prev, self.num_icons], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            W = tf.get_variable("W", shape=[prev, self.num_icons], initializer=tf.contrib.layers.xavier_initializer())
#             B = tf.get_variable("B", shape=[self.num_icons], initializer=tf.constant_initializer(0.1))
            W = tf.layers.dropout(inputs=W, rate=self.dropout, training=self.trainFlag, name="afterDropout")
            self.logits = tf.matmul(score, W)
#             self.regularizer += tf.nn.l2_loss(W) 
#             self.logits = tf.nn.sigmoid(tf.matmul(score, tf.nn.dropout(W,1-self.model_params.dropout)))
            Wmat.append(W)
            Bmat.append(B)
            print(Wmat, Bmat)
            
        
        print(self.logits, self.y)
#         self.loss_org = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.loss_org = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y)
#         self.loss_org = (self.logits - self.y)**2
#         self.loss = tf.reduce_mean(self.loss_org)
#         print(self.loss_org)
#         self.supp = tf.placeholder(tf.float32, shape = [None, self.num_icons], name = "supp")
#         self.loss = tf.multiply(self.loss_org, self.supp)
        self.loss = tf.reduce_mean(self.loss_org)#+0.0001*self.regularizer
#         print(self.loss)
        
    
    
    # train the model using the appropriate parameters
    def train(self):
        """Train the model"""
#         minimization_op = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(self.loss_org)
        minimization_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#         minimization_op = tf.train.GradientDescentOptimizer(self.model_params.learning_rate).minimize(self.loss)
        
        self.initializeSession()
        epoch = 0

        max_res = {"min1000":[0,0], "min5000":[0,0], "minWordAll":[0,0], "train1000": [0,0], "train5000": [0,0]} 
        while epoch < self.max_epochs:   
            training_idx = np.random.randint(len(self.trainset[0]), size=self.batch_size)
#             y_org = self.trainset[0][training_idx]
#             y_sup = self.suppress(y_org)
#             print(len(self.trainset[1][training_idx][0]))
#             print(self.trainset[0][training_idx][0])
#             print(y_sup[0])
#             raise
            
            _, current_loss, y, logits = self.session.run([minimization_op, self.loss, self.y,self.logits], feed_dict={
                self.phrase_vec:self.trainset[1][training_idx],
                self.y:self.trainset[0][training_idx],
                self.trainFlag: True
#                 self.supp:y_sup #np.array([[1]*self.num_icons]*self.model_params.batch_size)
            })
            
            if epoch <= 5000: 
                spe = 1000
            else:
                spe = 10
            
            if epoch % spe == 0:
                print("Epoch=%d loss=%8.5f" %(epoch, current_loss))
                print(max_res["minWordAll"]) 
#                 print(y[0], logits[0], loss_org[0])
                epoch += 1
                trainres = self.cal_top_n(self.trainset, "train train", N=2, stop = 10000)
                devres = self.cal_top_n(self.testset, "train dev1000", N=2,stop=5000)
#                 testres = self.cal_top_n(self.test, "train test", N=2,stop=2000)
                if not devres:
                    continue
                if devres[1] < max_res["min1000"][1]-0.01:
                    continue
                if (devres[1] > 0.2 and devres[1]>=max_res["min1000"][1] + 0.00) or devres[1]> 0.243:
                    if devres[1] > max_res["min1000"][1]:
                        max_res["min1000"] = devres
                    testres = self.cal_top_n(self.testset, "train devMinAll  ", N=2, stop = 116000)
#                     V = self.session.run(self.V[0])
                    
# #                     self.saveModel(self.model_params.model_folder("minworddev", devres[0], devres[1]),V)
                    
                    if testres[1] > max_res["minWordAll"][1]:
                        max_res["minWordAll"] = testres
#                         max_res["min5000"] = testres
#                         testminallres = self.cal_top_n(self.benchmarkDatasetMin, "testMinALL  ", N=2)
#                         if testminallres[1] > max_res["minWordAll"][1]:
# #                             testallres = self.cal_top_n(self.benchmarkDataset, "testALL  ", N=2)
# #                             if testminallres[1] > max_res["minWordAll"][1]:
#                             max_res["minWordAll"] = testminallres
# #                                 max_res["notMinAll"] = testallres
# #                             self.saveModel(self.model_params.model_folder("minword", testminallres[0], testminallres[1]), V)
                
            epoch += 1

        print("results when max dev accu:")
        print(max_res)        
        return max_res 
	
	
    # find top N icon indices and return P,R,F1,TP,TN,FP,FN
    def cal_top_n(self, dataset, str, N=2, stop = sys.maxsize):
        res  = self.session.run(self.logits, feed_dict={
            self.trainFlag: False,
            self.phrase_vec:dataset[1][:min(stop, sys.maxsize)],
        })
#         print(res,res.shape)
        results = [sorted(range(0, self.num_icons), key=lambda j:res[i][j], reverse=True)[:N] for i in range(min(stop, len(dataset[0])))]
#         print(res[0][results[0]], results[0])
        return self.cal_metrics(results, dataset[0][:min(stop, sys.maxsize)], str, N=N)
        
    
    # calculate details
    def cal_metrics(self, results, icons, str, N=2):
        # results: top N icon indices returned by Text2Vec for each phrase
        # icon idx for each phrase-icon pair
#         print(results)
#         print(icons)

#         if len(results) != len(labels) or len(results) != len(icons):
#             print("error: len of inputs not equal")
#             raise
        P, T, F = [-404]*N, [0]*N, [0]*N
        for i in range(len(results)):
            if str[:5] != "train":
                for n in range(N):
                    if icons[i] in results[i][:n+1]:
                        T[n] += 1
                    else:
                        F[n] += 1
            else:
#                 print(len(icons[i]))
                if icons[i][results[i][0]] == 1:
                    T[0] += 1
                    T[1] += 1
                elif icons[i][results[i][1]] == 1:
                    F[0] += 1
                    T[1] += 1
                else:
                    F[0] += 1
                    F[1] += 1
#                 print(icons[i][results[i][0]],  icons[i][results[i][1]], T, F)
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

        
        
        
M = Text2VecMulti()
M.train()