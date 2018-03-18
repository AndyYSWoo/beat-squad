
from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
from tqdm import tqdm


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn
from qa_model import QAModel
from vocab import _PAD, _UNK, _START_VOCAB, PAD_ID, UNK_ID

logging.basicConfig(level=logging.INFO)

def get_glove_char(glove_char_path, glov_char_dim):
    print "Loading GLoVE Char vectors from file: %s" % glove_char_path
    glove_char_size = 94
    char_emb_matrix = np.zeros((glove_char_size + len(_START_VOCAB), glov_char_dim))
    char2id = {}
    id2char = {}
    idx = 0
    for char in _START_VOCAB:
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    with open(glove_char_path, 'r') as fh:
        for line in tqdm(fh, total=glove_char_size):
            line = line.lstrip().rstrip().split(" ")
            char = line[0]
            vector = list(map(float, line[1:]))
            char_emb_matrix[idx, :] = vector
            char2id[char] = idx
            id2char[idx] = char
            idx += 1

    return char_emb_matrix, char2id, id2char

class CharModel(QAModel):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        self.char_emb_matrix , self.char2id, self.id2char =get_glove_char(FLAGS.glove_char_path, FLAGS.char_embedding_size)
        super(CharModel, self).__init__(FLAGS, id2word, word2id, emb_matrix)
    # override to add char embedding placeholders
    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Chars
        self.context_char_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.char_limit])
        # self.context_char_masks = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.char_limit])
        self.qn_char_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.char_limit])
        # self.qn_char_masks = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.char_limit])

    # override to add char embedding
    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)

            # Chars
            char_embedding_matrix = tf.constant(self.char_emb_matrix, dtype=tf.float32, name="char_emb_matrix")  # shape (96, char_embedding_size)
            self.char_emb_matrix = None # release memory, hopefully
            self.context_char_embs = tf.reshape(embedding_ops.embedding_lookup(char_embedding_matrix, self.context_char_ids),
                                                [-1, self.FLAGS.char_limit, self.FLAGS.char_embedding_size]) # shape (batch_size * context_len, char_limit, embedding_size)
            self.qn_char_embs = tf.reshape(embedding_ops.embedding_lookup(char_embedding_matrix, self.qn_char_ids),
                                                [-1, self.FLAGS.char_limit, self.FLAGS.char_embedding_size]) # shape (batch_size * question_len, char_limit, embedding_size)
    # override to add char encoder
    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        context_input_lens = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.context_char_ids, tf.bool), tf.int32), axis=2), [-1])
        qn_input_lens = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qn_char_ids, tf.bool), tf.int32), axis=2), [-1])
        cell_fw = rnn_cell.GRUCell(self.FLAGS.hidden_size)
        cell_bw = rnn_cell.GRUCell(self.FLAGS.hidden_size)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.context_char_embs, context_input_lens, dtype=tf.float32)
        ch_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [-1, self.FLAGS.context_len, 2 * self.FLAGS.hidden_size])
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.qn_char_embs, qn_input_lens, dtype=tf.float32)
        qh_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [-1, self.FLAGS.question_len, 2 * self.FLAGS.hidden_size])

        self.context_embs = tf.concat([self.context_embs, ch_emb], axis=2)
        self.qn_embs = tf.concat([self.qn_embs, qh_emb], axis=2)

        # ToDo Deep encoder
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)

    # override to add char to batch
    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout  # apply dropout

        add_chars_to_feed(input_feed, self.context_char_ids, self.qn_char_ids, batch, self.id2word, self.char2id, self.FLAGS.char_limit)

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout
        add_chars_to_feed(input_feed, self.context_char_ids, self.qn_char_ids, batch, self.id2word, self.char2id, self.FLAGS.char_limit)

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout
        add_chars_to_feed(input_feed, self.context_char_ids, self.qn_char_ids, batch, self.id2word, self.char2id, self.FLAGS.char_limit)

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


def add_chars_to_feed(input_feed, c_ph, q_ph, batch, id2word, char2id, char_limit):
    context_char_ids = word_ids_to_char_ids(batch.context_ids, id2word, char2id, char_limit)
    qn_char_ids = word_ids_to_char_ids(batch.qn_ids, id2word, char2id, char_limit)
    input_feed[c_ph] = context_char_ids
    input_feed[q_ph] = qn_char_ids

def word_ids_to_char_ids(word_ids, id2word, char2id, char_limit):
    batch_size, sentence_len = word_ids.shape
    char_ids = np.zeros((batch_size, sentence_len, char_limit))
    for batch in range(batch_size):
        for sentence in range(sentence_len):
            word_id = word_ids[batch][sentence]
            if word_id == UNK_ID: continue
            if word_id == PAD_ID: break
            word = id2word[word_id]
            for i in range(0, min(char_limit, len(word))):
                char_ids[batch][sentence][i] = char2id.get(word[i], UNK_ID)
    return char_ids