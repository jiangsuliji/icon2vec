#!/usr/bin/env python
"""icon2vec model implemented in TensorFlow.

File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
Details of the model can be found below.
Based on Ben Eisner, Tim Rocktaschel's good work. 
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


class Text2Vec:
    """Class for representing the model in TensorFlow."""

    # define the model
    def __init__(self, model_params, num_icons, trainset, devset, testset):
        """Constructor for the Text2Vec model

        Args:
            model_params: Parameters for the model
            num_icons: Number of icons we will ultimately train
        """
        self.model_params = model_params
        self.num_cols = num_icons
        
        self.initializeDataset(trainset, devset, testset)
        self.initializeDatasetWithBenchmarkTraining()
        self.__open_benchmark()
#         raise 
        
        # row - phrase input to the graph
        self.phrase_vec = tf.placeholder(tf.float32, shape=[None, model_params.in_dim], name='phrase_vec')
            
        # Correlation between an icon and a phrase
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

        # Icon indices in current batch
        self.col = tf.placeholder(tf.int32, shape=[None], name='col')
        
        # 1st pass: init V embeddings for icons - multi layers
        # Column embeddings (here icon representations)
        self.V = []
        for idx, num in enumerate(model_params.nn_params):
            V = tf.get_variable("V"+str(idx), shape=[num_icons, num], initializer=tf.contrib.layers.xavier_initializer())  
#             V = tf.Variable(tf.random_uniform([num_icons, num], -0.1, 0.1), name="V"+str(idx))  
            self.V.append(V)
#         print(self.V)
                
        # 2nd pass: calculate score, build projection matrix P between V layers
        prev = model_params.in_dim
        score = self.phrase_vec
        layers = len(model_params.nn_params)
        for idx in range(layers):
            if model_params.in_dim != model_params.nn_params[0] or idx > 0:
#                 P = tf.Variable(tf.random_uniform([prev, model_params.nn_params[idx]]), name="P"+str(idx))
                P = tf.get_variable("P"+str(idx), shape=[prev, model_params.nn_params[idx]], initializer=tf.contrib.layers.xavier_initializer())
                prev = model_params.nn_params[idx]
                score = tf.matmul(score, P)
#                 score = tf.tanh(tf.matmul(score, P))
#                 print("\n")
#                 print("P",P)
#                 print("score", score)
            
            V = tf.nn.embedding_lookup(self.V[idx], self.col)
#             print("V",V)
            if idx != layers - 1:
                score = tf.multiply(score, V)
            else:
                score = tf.multiply(score, tf.nn.dropout(V, (1-model_params.dropout)))
#             print("score", score)
#             print("\n")
#         print(score)
        
    	# custom loss
        self.score = tf.reduce_sum(score, 1)

        # Probability of match
        self.prob = tf.sigmoid(self.score)
        # Calculate the cross-entropy loss
#         self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.score, logits=self.y, pos_weight=1)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.y) 
        
    def initializeDataset(self, trainset, devset, testset):
        icon_idx, phrase_embedding, labels = [], [], []
        ph_idx = []
        for item in trainset:
            icon_idx.append(item[0])
            phrase_embedding.append(item[1])
            labels.append(item[2])
        self.trainset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
        
        icon_idx, phrase_embedding, labels = [], [], []
        for item in devset:
            icon_idx.append(item[0])
            phrase_embedding.append(item[1])
            labels.append(item[2])
        self.devset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
        
        icon_idx, phrase_embedding, labels = [], [], []
        for item in testset:
            icon_idx.append(item[0])
            phrase_embedding.append(item[1])
            labels.append(item[2])
        self.testset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
        
    def initializeDatasetWithBenchmarkTraining(self):
        """initialize benchmark """ 
        fileObject = open("data/benchmarks/trainset_12-2017_9-1-2018_025Unk.ss.csv.glove.p", 'rb')
        benchmarktrainraw = pk.load(fileObject)
        fileObject.close()
        # parse the positive
        icon_idx, phrase_embedding, labels = [], [], []
        ph_idx = []
        for item in benchmarktrainraw:
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(1)
        # generate negative
        half = len(labels)
        for i in range(half):
            icon_idx.append((icon_idx[i]+np.random.randint(0,1000))%self.num_cols)
            phrase_embedding.append(phrase_embedding[i])
            labels.append(0)
        # combine
        self.trainset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]
        
        
    def __open_benchmark(self):
        fileObject = open("data/benchmarks/testset_SingleIcon_9-1_10-22-2018_025Unk.ss.csv.glove.p", 'rb')
        self.benchmark = pk.load(fileObject)
        fileObject = open("data/benchmarks/testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv.glove.p", 'rb')
        self.benchmarkMin = pk.load(fileObject)
#         print(len(self.benchmark), self.benchmark[0])
        fileObject.close()
        
        icon_idx, phrase_embedding, labels, icon, phrase = [], [], [], [], []
        for item in self.benchmark:
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(1)
            icon.append(item[2])
            phrase.append(item[3])
        self.benchmarkDataset = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels), np.array(icon), np.array(phrase)]
        
        icon_idx, phrase_embedding, labels = [], [], []
        for item in self.benchmarkMin:
            icon_idx.append(item[1])
            phrase_embedding.append(item[0])
            labels.append(1)
        self.benchmarkDatasetMin = [np.array(icon_idx), np.array(phrase_embedding), np.array(labels)]   
        
        
    def initializeSession(self):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)
    
    
    # train the model using the appropriate parameters
    def train(self):
        """Train the model on a given knowledge base"""
        self.optimizer = tf.train.AdamOptimizer(self.model_params.learning_rate)
        minimization_op = self.optimizer.minimize(self.loss)
        
        self.initializeSession()
        epoch = 0
        total_data_entry = self.trainset[0].shape[0]
        half_data_entry = self.trainset[0].shape[0]//2
        max_accuracy_top2 = [[0,0],[0,0]] 
        while epoch < self.model_params.max_epochs:   
#             print(total_data_entry, half_data_entry)
            training_idx_pos = np.random.randint(half_data_entry, size=self.model_params.batch_size//2)
            training_idx_neg = np.random.randint(half_data_entry, size=self.model_params.batch_size//2)
            training_idx_neg = [i+ half_data_entry for i in training_idx_neg]
            training_idx = np.concatenate((training_idx_pos, training_idx_neg),axis =0)
            shuffle(training_idx)
#             y = [1 if i < half_data_entry else 0 for i in training_idx]
            y = [self.trainset[2][i] for i in training_idx]
            
#             print(training_idx_neg, training_idx_pos)
#             print("===",training_idx)
#             print(y)
        
            _, current_loss = self.session.run([minimization_op, self.loss], feed_dict={
                self.col:self.trainset[0][training_idx],
                self.phrase_vec: self.trainset[1][training_idx],
                self.y:np.array(y)
            })
            current_loss = sum(current_loss)

            print("Epoch=%d loss=%3.1f" %(epoch, current_loss))
            if epoch % 10 == 0:
                epoch += 1
#                 devres = self.cal_top_n(self.devset, "dev      ", N=2)
                devres = self.cal_top_n(self.benchmarkDatasetMin, "dev1000 ", N=2,stop=1000)
                if not devres:
                    continue
                if devres[1] < max_accuracy_top2[0][1]:
                    continue
                testres = self.cal_top_n(self.benchmarkDatasetMin, "test5000  ", N=2, stop=5000)
#                 testres = self.cal_top_n(self.testset, "test     ", N=2)
                max_accuracy_top2 = [devres, testres]
#                 if devres[1] > 0.15:
#                     benchmarkres = self.cal_top_n(self.benchmarkDataset, "bench   ", N=2, stop=1000) 
#                     benchmarkminires = self.cal_top_n(self.benchmarkDatasetMin, "benchmin", N=2)
#                     max_accuracy_top2 = [devres, testres, benchmarkres]#, benchmarkminres]
            epoch += 1

        print("results when max dev accu:")
        print(max_accuracy_top2)
        self.print_top_accuracy(max_accuracy_top2[0],"dev") 
        self.print_top_accuracy(max_accuracy_top2[1],"test") 
        
        return max_accuracy_top2
    
    # find top N icon indices and return P,R,F1,TP,TN,FP,FN
    def cal_top_n(self, dataset, str, N=2, stop = sys.maxsize):
        # quick assessment for early termination:
        if len(dataset) > 20000:
            res = self.cal_top_n(dataset, str+"fast", N=2, stop = stop)
            return res
        
        results = [] # for phrase - top N icons
        indices_arr = range(0, self.num_cols)
        for ph_idx in range(min(stop, len(dataset[1]))):
#             print(dataset[1][ph_idx])
#             print(np.tile(np.array(dataset[1][ph_idx]), (self.num_cols, 1)).shape)
            res = self.session.run(self.prob, feed_dict={
                    self.col:indices_arr,
                    self.phrase_vec: np.tile(np.array(dataset[1][ph_idx]), (self.num_cols, 1))
            })

            results.append(sorted(indices_arr, key=lambda i:res[i], reverse=True)[:N])
        if str == "bench   ":
            for i in range(10):
                print("phrase:", dataset[4][i])
                print("correct icon:",self.model_params.mp_idx2name[dataset[0][i]], \
                      "suggested icons(Top2)", self.model_params.mp_idx2name[results[i][0]],self.model_params.mp_idx2name[ results[i][1]])
                print("\n")
        return self.cal_metrics(results, dataset[2], dataset[0], str, N=N)
        
        
    # calculate details
    def cal_metrics(self, results, labels, icons, str, N=2):
        # results: top N icon indices returned by Text2Vec for each phrase
        # label for each phrase-icon pair
        # icon idx for each phrase-icon pair
#         print(results)
#         print(labels)
#         print(icons)

#         if len(results) != len(labels) or len(results) != len(icons):
#             print("error: len of inputs not equal")
#             raise
        P, T, F = [-404]*N, [0]*N, [0]*N
        for i in range(len(results)):
            if labels[i] == 1.0:
                for n in range(N):
                    if icons[i] in results[i][:n+1]:
                        T[n] += 1
                    else:
                        F[n] += 1
        for n in range(N):
            P[n] = T[n]/(T[n]+F[n])
            
        self.print_top_accuracy_TP(P, T, F, str)
        return P
    
    # train set evaluation
    # TODO: update
    def test_on_train(self):
        res = self.session.run(self.prob, feed_dict={
                self.col:self.trainset[0],
                self.phrase_vec: self.trainset[1]
        })
        y_pred = [1 if y > self.model_params.class_threshold else 0 for y in res]
        return y_pred, self.trainset[2]
       
    
    def print_top_accuracy_TP(self, P, T, F, st):
        if len(P) != len(T) or len(T) != len(F):
            raise
        s = "\t"+st + "\t"
        for i in range(len(P)):
            s += "P" +str(i+1)+"="
            s += "%3.2f," %(P[i])
        s = s[:-1] + "; "
        for i in range(len(T)):
            s += "T"+str(i+1)+"="+str(T[i]) + ",F" + str(i+1)+"="+str(F[i])+","
        s = s[:-1]
        print(s)
    
    def print_top_accuracy(self,P,st):
        s = "\t"+st+"\t"
        for i in range(len(P)):
            s+= "P" +str(i+1)+"="
            s+= "%3.2f," %(P[i])
        s = s[:-1]
        print(s)
        
    def close(self):
        self.session.close()
        ops.reset_default_graph()
        tf.reset_default_graph()

        