#!/usr/bin/env python
"""icon2vec model implemented in TensorFlow.

File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
Details of the model can be found below.
Based on Ben Eisner, Tim Rocktaschel's good work. We will work on a quite different approach later but some of the parser codes are written based on their codes. 
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


class ModelParams:
    """Convenience class for passing around model parameters"""

    def __init__(self, in_dim, max_epochs, batch_size, learning_rate, dropout, class_threshold, nn_params):
        """Create a struct of all parameters that get fed into the model

        Args:
            in_dim: Dimension of the word vectors supplied to the algorithm (e.g. word2vec)
            max_epochs: Max number of training epochs
            learning_rate: Learning rate
            dropout: Dropout rate
            class_threshold: Classification threshold for accuracy
        """
        self.class_threshold = class_threshold
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.in_dim = in_dim
        self.nn_params = nn_params
        # load icon idx map first
        fileObject = open("data/iconIndex2NameMap.p", 'rb')
        self.mp_idx2name = pk.load(fileObject)
#         print(self.mp_icon2idx)
        fileObject.close()

    # TODO jili5: update this later
    def model_folder(self, dataset_name, p1, p2):
        """Get the model path for a given dataset

        Args:
            dataset_name: The name of the dataset we used to generate training data

        Returns:
            The model path for a given dataset
        """
        return str.format(
            str.format('./results/{}/lr-{}_ep-{}_dr-{}_P1-{}_P2-{}', dataset_name, self.learning_rate, self.max_epochs, int(self.dropout * 10), p1, p2))
