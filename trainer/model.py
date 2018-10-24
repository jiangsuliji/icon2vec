#!/usr/bin/env python
"""icon2vec model implemented in TensorFlow.

File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
Details of the model can be found below.
Based on Ben Eisner, Tim Rocktaschel's work. 
"""

# External dependencies
import tensorflow as tf
import numpy as np
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
        self.initializeDataset(trainset, devset, testset)
        
        self.model_params = model_params
        self.num_cols = num_icons
        
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
#             V = tf.get_variable("V"+str(idx), shape=[num_icons, num], initializer=tf.contrib.layers.xavier_initializer())  
            V = tf.Variable(tf.random_uniform([num_icons, num], -0.1, 0.1), name="V"+str(idx))  
            self.V.append(V)
#         print(self.V)
                
        # 2nd pass: calculate score, build projection matrix P between V layers
        prev = model_params.in_dim
        score = self.phrase_vec
        layers = len(model_params.nn_params)
        for idx in range(layers):
            if model_params.in_dim != model_params.nn_params[0] or idx > 0:
                P = tf.get_variable("P"+str(idx), shape=[prev, model_params.nn_params[idx]], initializer=tf.contrib.layers.xavier_initializer())
                prev = model_params.nn_params[idx]
                score = tf.tanh(tf.matmul(score, P))
#                 print("\n")
#                 print("P",P)
#                 print("score", score)
            
            V = tf.nn.embedding_lookup(self.V[idx], self.col)
#             print("V",V)
            if idx != layers - 1:
                score = tf.multiply(score, V)
            else:
                score = tf.multiply(tf.nn.dropout(score, (1-model_params.dropout)), V)
#             print("score", score)
#             print("\n")
#         print(score)
        
    	# custom loss
        self.score = tf.reduce_sum(score, 1)

        # Probability of match
        self.prob = tf.sigmoid(self.score)

        # Calculate the cross-entropy loss
        self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.score, logits=self.y, pos_weight=1)
        
        
    def initializeDataset(self, trainset, devset, testset):
        icon_idx, phrase_embedding, labels = [], [], []
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
        max_accuracy = [[0],[0],[0]]
        while epoch < self.model_params.max_epochs:   
#             print(total_data_entry, half_data_entry)
            training_idx_pos = np.random.randint(half_data_entry, size=self.model_params.batch_size//2)
            training_idx_neg = np.random.randint(half_data_entry, size=self.model_params.batch_size//2)
            training_idx_neg = [i+ half_data_entry for i in training_idx_neg]
            training_idx = np.concatenate((training_idx_pos, training_idx_neg),axis =0)
            shuffle(training_idx)
            y = [0 if i < half_data_entry else 1 for i in training_idx]
#             y = [self.trainset[2][i] for i in training_idx]
            
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
            devres = self.cal_top_n(self.devset, "dev", N=2)
            testres = self.cal_top_n(self.testset, "test", N=2)
            
            if devres and testres and devres[0] > max_accuracy[0][0]:
                max_accuracy = [devres, testres]
            epoch += 1
            
        print("results when max dev accu:")
        print(max_accuracy)
        
    
    # find top N icon indices and return P,R,F1,TP,TN,FP,FN
    def cal_top_n(self, dataset, str, N=2):
        results = [] # for phrase - top N icons
        indices_arr = range(0, self.num_cols)
        for ph_idx in range(len(dataset[1])):
#             print(dataset[1][ph_idx])
#             print(np.tile(np.array(dataset[1][ph_idx]), (self.num_cols, 1)).shape)
            res = self.session.run(self.prob, feed_dict={
                    self.col:indices_arr,
                    self.phrase_vec: np.tile(np.array(dataset[1][ph_idx]), (self.num_cols, 1))
            })
#             print(res)
            rtn_0 = sorted(indices_arr, key=lambda i:res[i], reverse=True)[:N]
            rtn_1 = []
            if res[rtn_0[0]] >= self.model_params.class_threshold:
                rtn_1.append(rtn_0[0])
            if res[rtn_0[1]] >= self.model_params.class_threshold:
                rtn_1.append(rtn_0[1])
#             print(res[rtn_1[0]], res[rtn_1[1]], sorted(res, reverse=True)[:N])
#             print(dataset[0][ph_idx] in sorted(res, reverse=True)[:4])
            results.append(rtn_1)
        return self.cal_metrics(results, dataset[2], dataset[0], str)
        
        
    # calculate details
    def cal_metrics(self, results, labels, icons, str):
        # results: top N icon indices returned by Text2Vec for each phrase
        # label for each phrase-icon pair
        # icon idx for each phrase-icon pair
#         print(results)
#         print(labels)
#         print(icons)
        if len(results) != len(labels) or len(results) != len(icons):
            print("error: len of inputs not equal")
            raise
        T1, F1, T2, F2 = 0, 0, 0, 0
        for i in range(len(results)):
            if icons[i] in results[i]:
                T2 += 1
            else:
                F2 += 1
            if icons[i] == results[i][0]:
                T1 += 1
            else:
                F1 += 1
        accuracy1, accuracy2 = T1/(T1+F1), T2/(T2+F2)
        print("  %s\taccuracy1=%3.2f, accuracy2=%3.2f; T1=%d,F1=%d,T2=%d,F2=%d" 
              %(str,accuracy1, accuracy2, T1, F1, T2, F2))
        return [accuracy1, accuracy2, T1, F1, T2, F2]
    
    
    # train set evaluation
    def test_on_train(self):
        res = self.session.run(self.prob, feed_dict={
                self.col:self.trainset[0],
                self.phrase_vec: self.trainset[1]
        })
        y_pred = [1 if y > self.model_params.class_threshold else 0 for y in res]
        return y_pred, self.trainset[2]
       
        