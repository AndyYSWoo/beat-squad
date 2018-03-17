
from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn
from qa_model import QAModel

logging.basicConfig(level=logging.INFO)

def get_glove_char():
    pass


class CharModel(QAModel):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):

        super(CharModel, self).__init__(FLAGS, id2word, word2id, emb_matrix)
