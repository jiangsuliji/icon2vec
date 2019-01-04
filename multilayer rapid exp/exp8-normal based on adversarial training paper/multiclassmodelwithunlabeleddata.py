#!/usr/bin/env python
"""
File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
"""

# External dependencies
import tensorflow as tf
# from tensorflow.python.framework import ops
import numpy as np
import sys
from os import environ
from random import shuffle
from collections import namedtuple, defaultdict

# import sklearn.metrics as metrics
# from warnings import filterwarnings 
# filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim.models as gs
import pickle as pk
import time
import train_utils
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable all debugging logs
tf.logging.set_verbosity(tf.logging.FATAL)

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"


Document = namedtuple('Document', \
    'icon    icon_idx  label\
     collection    collection_idx\
     phrase  phrase_vec   norm_phrase_vec\
     is_validation is_test\
    ')

#-----------------------------------------------------------------------
#                  multi class classifier
#-----------------------------------------------------------------------
class Text2VecMulti:
    """class for multi class model"""
    def __init__(self):
        self.num_icons = 844
        self.batch_size = 650
        self.dropout = 0.2
        self.max_epochs = 30000000
        self.learning_rate = 0.0003
        self.nn_params = [300, 1000]
        self.in_dim = 300
        self.s = time.time()
        self.initializeDataset()
        self.initializeModel()

    def print_time(self, s):
        print(s, "time=",time.time()-self.s )
        self.s = time.time()
        
    def initializeDataset(self):
        fileObject = open("tmp/test.fasttext.multiclass.p", 'rb')
        benchmarktrainraw = pk.load(fileObject)
        icon_idx, phrase_embedding, labels, phrase = [], [], [], []
        oldicon, newicon = 0, 0
        for doc in benchmarktrainraw:
            oldicon += 1
            for ii in doc.icon_idx:
              if ii > 490:
                newicon += 1
                oldicon -= 1
                break
            phrase_embedding.append(doc.norm_phrase_vec)
            labels.append(np.array(doc.label))
            phrase.append(doc.phrase)
            icon_idx.append(doc.icon)

        icon_idx, phrase_embedding, labels, phrase = np.array(icon_idx), np.array(phrase_embedding), np.array(labels), np.array(phrase)
        self.testset = [phrase_embedding, labels, phrase, icon_idx]
        print("Testset-old,new,percentage of new:", oldicon, newicon, newicon/(newicon+oldicon))

        fileObject = open("tmp/description_test.fasttext.multiclass.p", 'rb')
        benchmarktrainraw = pk.load(fileObject)
        icon_idx, phrase_embedding, labels, phrase = [], [], [], []
        oldicon, newicon = 0, 0
        for doc in benchmarktrainraw:
            oldicon += 1
            for ii in doc.icon_idx:
              if ii > 490:
                newicon += 1
                oldicon -= 1
                break
            phrase_embedding.append(doc.norm_phrase_vec)
            labels.append(np.array(doc.label))
            phrase.append(doc.phrase)
            icon_idx.append(doc.icon)

        icon_idx, phrase_embedding, labels, phrase = np.array(icon_idx), np.array(phrase_embedding), np.array(labels), np.array(phrase)
        self.description_test = [phrase_embedding, labels, phrase, icon_idx]
        print("description_test-old,new,percentage of new:", oldicon, newicon, newicon/(newicon+oldicon))

        fileObject = open("tmp/train.fasttext.multiclass.p", 'rb')
        benchmarktrainraw = pk.load(fileObject)
        icon_idx, phrase_embedding, labels, phrase = [], [], [], []
        icon_idx1, phrase_embedding1, labels1, phrase1 = [], [], [], []
        oldicon, newicon = 0, 0
        for doc in benchmarktrainraw:
            oldicon += 1
            phrase_embedding.append(doc.norm_phrase_vec)
            labels.append(np.array(doc.label))
            phrase.append(doc.phrase)
            icon_idx.append(doc.icon)
            for ii in doc.icon_idx:
              if ii > 490:
                newicon += 1
                oldicon -= 1
                phrase_embedding1.append(doc.norm_phrase_vec)
                labels1.append(np.array(doc.label))
                phrase1.append(doc.phrase)
                icon_idx1.append(doc.icon)
                
                icon_idx.pop()
                phrase_embedding.pop()
                labels.pop()
                phrase.pop()
                break
            
        icon_idx, phrase_embedding, labels, phrase = np.array(icon_idx), np.array(phrase_embedding), np.array(labels), np.array(phrase)
        icon_idx1, phrase_embedding1, labels1, phrase1 = np.array(icon_idx1), np.array(phrase_embedding1), np.array(labels1), np.array(phrase1)

        self.trainset = [phrase_embedding, labels] #, phrase, icon_idx]
        self.trainset1 = [phrase_embedding1, labels1] #, phrase1, icon_idx1]

        # parse train_unlabel dataset
        self.unlabel_batch_num = 0
        self.train_unlabel = []
        # self.fetchNextBatchUnlabeledData()
        fileObject.close()
        print("parsed train/test/train_unlabel:", len(self.trainset[0]), len(self.testset[0]), len(self.train_unlabel))
        print("Trainset-old,new,percentage of new:", oldicon, newicon, newicon/(newicon+oldicon))
        print("Trainset-old,new:", len(self.trainset[0]), len(self.trainset1[0]))
        self.print_time("initializeDataset load")

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
        self.scores = tf.nn.softmax(logits=self.logits)
        
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

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=100)
    
    
    # train the model using the appropriate parameters
    def train(self):
        """Train the model"""
#         minimization_op = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(self.loss_org)
        minimization_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#         minimization_op = tf.train.GradientDescentOptimizer(self.model_params.learning_rate).minimize(self.loss)
        
        self.initializeSession()
        epoch = 0
        self.unlabeled_passes = 0
        max_res = {"min1000":[0,0], "min5000":[0,0], "minWordAll":[0,0], "train1000": [0,0], "train5000": [0,0]} 
        while epoch < self.max_epochs:
            training_idx = np.random.randint(len(self.trainset[1]), size=int(self.batch_size*0.5))
            training_idx1 = np.random.randint(len(self.trainset1[1]), size=int(self.batch_size*0.5))
            feed_phrase_vec = np.concatenate((self.trainset[0][training_idx],self.trainset1[0][training_idx1]), axis=0)
            feed_y = np.concatenate((self.trainset[1][training_idx], self.trainset1[1][training_idx1]), axis=0)
            feed_idx = np.random.randint(len(feed_phrase_vec), size=len(feed_phrase_vec))
            # print(self.trainset[0][training_idx][0])
            # print(len(self.trainset[1][training_idx][0]))
            _, current_loss, y, logits = self.session.run([minimization_op, self.loss, self.y,self.logits], feed_dict={
                self.phrase_vec:feed_phrase_vec[feed_idx],
                self.y:feed_y[feed_idx],
                self.trainFlag: True
            })
            
            if epoch <= 26000:
                spe = 2000
            else:
                spe = 500*(1+self.unlabeled_passes//10)**2
            
            if epoch % spe == 0:
                print("Epoch=%d loss=%8.5f" %(epoch, current_loss))
                print(max_res["minWordAll"]) 
#                 print(y[0], logits[0], loss_org[0])
                epoch += 1
                trainres = self.cal_top_n(self.trainset, "train train", fNewIconFireRate=False, N=2, stop = 10000)
                devres = self.cal_top_n(self.testset, "train dev1000", fNewIconFireRate=False, N=2,stop=1000)
                tt = self.cal_top_n(self.description_test, "train description", fNewIconFireRate=True, N=2)
                if not devres:
                    continue
                if devres[1] < max_res["min1000"][1]-0.01:
                    continue
                if (devres[1] > 0.2 and devres[1]>=max_res["min1000"][1] - 0.25) or devres[1]> 0.243:
                    if devres[1] > max_res["min1000"][1]:
                        max_res["min1000"] = devres
                    testres = self.cal_top_n(self.testset, "train devMinAll  ", fNewIconFireRate=True, N=2, stop = 116052)
                    
                    
                    if testres[1] > max_res["minWordAll"][1]:
                        max_res["minWordAll"] = testres

                        # save_path = self.saver.save(self.session, "model/"+str(testres[1]))
                        
#                         max_res["min5000"] = testres
#                         testminallres = self.cal_top_n(self.benchmarkDatasetMin, "testMinALL  ", N=2)
#                         if testminallres[1] > max_res["minWordAll"][1]:
# #                             testallres = self.cal_top_n(self.benchmarkDataset, "testALL  ", N=2)
# #                             if testminallres[1] > max_res["minWordAll"][1]:
#                             max_res["minWordAll"] = testminallres
# #                                 max_res["notMinAll"] = testallres
                
                # if max_res["minWordAll"][1] > 0.17:
                  # self.runUnlabeledData()

            epoch += 1

        print("results when max dev accu:")
        print(max_res)
        return max_res 

    # find top N icon indices and return P,R,F1,TP,TN,FP,FN
    def cal_top_n(self, dataset, str, fNewIconFireRate=False, N=2, stop = sys.maxsize):
      res  = self.session.run(self.logits, feed_dict={
          self.trainFlag: False,
          self.phrase_vec:dataset[0][:min(stop, sys.maxsize)],
      })
#         print(res,res.shape)
      results = [sorted(range(0, self.num_icons), key=lambda j:res[i][j], reverse=True)[:N] for i in range(min(stop, len(dataset[0])))]
#         print(res[0][results[0]], results[0])
      if fNewIconFireRate:
          train_utils.cal_NewIconFireRate(results, str)
      return train_utils.cal_metrics(results, dataset[1][:min(stop, sys.maxsize)], str, N=N)
        
    def runUnlabeledData(self):
      def genLabel(a):
        res = [0]*self.num_icons
        res[a] = 1
        return res

      print("INFO::Unlabel::loaded batches=", self.unlabel_batch_num, "passes=", self.unlabeled_passes)
      low, high, high_old = min(0.05*(1+self.unlabeled_passes//10), 0.5), 0.95, 0.85
      res = self.session.run(self.scores, feed_dict={
          self.trainFlag: False,
          self.phrase_vec:np.array(self.train_unlabel),
      })
      res = res.tolist()
      # print(res)
      score, idx = [0]*len(res), [-1]*len(res)
      for i in range(len(res)):
        curr = 0.0
        for j, v in enumerate(res[i]):
            if v >= curr:
              score[i], idx[i] = v, j
              curr = v
        
      cntAdd, cntRemove = 0, 0
      train_unlabel_added = [[],[]]
      # print(res[0], len(res))
      # print(score[0], idx[0])
      for i in range(len(self.train_unlabel)-1, -1, -1):
        if score[i] < low:
          cntRemove += 1
          self.train_unlabel.pop(i)
          continue
        if score[i] > high_old and idx[i] < 491: 
            cntRemove += 1
            self.train_unlabel.pop(i)
            continue
        if score[i] > high and idx[i] >= 491:
            cntAdd += 1
            train_unlabel_added[0].append(self.train_unlabel.pop(i))
            train_unlabel_added[1].append(genLabel(idx[i]))

      print("INFO::Unlabel: Added/Removed/Left", cntAdd, cntRemove,len(self.train_unlabel))

      self.trainset1[0] = np.concatenate((self.trainset1[0], np.array(train_unlabel_added[0])), axis=0)
      self.trainset1[1] = np.concatenate((self.trainset1[1], np.array(train_unlabel_added[1])), axis=0)

      print("INFO::Unlabel::Old/New/NewIconRatio",  len(self.trainset[0]), len(self.trainset1[0]), len(self.trainset1[0])/(len(self.trainset[0])+len(self.trainset1[0])))
      self.unlabeled_passes += 1
      if len(self.train_unlabel) < 150000:
        self.fetchNextBatchUnlabeledData()
      
      
      

    def fetchNextBatchUnlabeledData(self):
      print("INFO::Unlabel::Loading batch", self.unlabel_batch_num)
      self.train_unlabel = []
      fileObject = open("tmp/train_unlabel.fasttext.multiclass.p."+str(self.unlabel_batch_num), 'rb')
      benchmarktrainraw = pk.load(fileObject)
      fileObject.close()
      for doc in benchmarktrainraw:
          self.train_unlabel.append(doc.norm_phrase_vec)
      print("INFO::Unlabel::Load Done. Left size=", len(self.train_unlabel))
      self.unlabel_batch_num += 1
      
      
M = Text2VecMulti()
M.train()