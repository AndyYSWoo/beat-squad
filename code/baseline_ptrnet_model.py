import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_model import QAModel
from modules import *
from rnet_ptrnet_modules import *
from tensorflow.python.ops import rnn_cell
import numpy as np

class BaselinePtrModel(QAModel):
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
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Output layer with pointer networks
        with tf.variable_scope('PointerNetwork'):
            with tf.variable_scope('PointerNetworkInitial'):
                V_Q = tf.get_variable('V_Q', shape=(1, context_hiddens.get_shape().as_list()[2]),
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)  # (1, hidden_size*2)
                initial_state_layer = MultiplicativeAttnWithParameterKey(self.keep_prob, self.FLAGS.hidden_size * 2,
                                                                         self.FLAGS.hidden_size * 2)
                _, initial_state = initial_state_layer.build_graph(question_hiddens, self.qn_mask,
                                                                   V_Q)  # (batch_size, 1, hidden_size*2)

            with tf.variable_scope('PointerNetworkLayerStart'):
                start_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS.hidden_size * 4)
                # (batch_size, 1, context_len), (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
                start_layer_masked_logits, start_layer_prob_dist, start_layer_output = start_layer.build_graph(
                    blended_reps, self.context_mask, initial_state)
                with vs.variable_scope("StartDist"):
                    start_layer_masked_logits = tf.squeeze(start_layer_masked_logits, [1])  # (batch_size, context_len)
                    start_layer_prob_dist = tf.squeeze(start_layer_prob_dist, [1])  # (batch_size, context_len)
                    self.logits_start = start_layer_masked_logits
                    self.probdist_start = start_layer_prob_dist

            cell = rnn_cell.GRUCell(self.FLAGS.hidden_size * 2)
            _, new_state = cell(tf.squeeze(start_layer_output, [1]),
                                tf.squeeze(initial_state, [1]))  # (batch_size, hidden_size*2)
            new_state = tf.expand_dims(new_state, 1)  # (batch_size, 1, hidden_size*2)

            with tf.variable_scope('PointerNetworkLayerEnd'):
                end_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS.hidden_size * 4)
                # (batch_size, 1, context_len), (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
                end_layer_masked_logits, end_layer_prob_dist, end_layer_output = end_layer.build_graph(blended_reps,
                                                                                                       self.context_mask,
                                                                                                       new_state)
                with vs.variable_scope("EndDist"):
                    end_layer_masked_logits = tf.squeeze(end_layer_masked_logits, [1])  # (batch_size, context_len)
                    end_layer_prob_dist = tf.squeeze(end_layer_prob_dist, [1])  # (batch_size, context_len)
                    self.logits_end = end_layer_masked_logits
                    self.probdist_end = end_layer_prob_dist


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Use dynamic programming to find maximum start_dist * end_dist where start_dist <= end_dist
        shape = start_dist.shape # (batch_size, context_len)
        start_pos = np.zeros(shape[0], dtype=np.int64)
        end_pos = np.zeros(shape[0], dtype=np.int64)

        for batch_index in range(shape[0]):
            max_start_dist_index = np.zeros(shape[1], dtype=np.int64)
            max_start_dist_index[0] = 0
            for start_dist_index in range(shape[1]):
                if start_dist[batch_index][start_dist_index] > \
                        start_dist[batch_index][max_start_dist_index[start_dist_index - 1]]:
                    max_start_dist_index[start_dist_index] = start_dist_index
                else:
                    max_start_dist_index[start_dist_index] = max_start_dist_index[start_dist_index - 1]

            for end_dist_index in range(shape[1]):
                current_max_product = start_dist[batch_index][start_pos[batch_index]] \
                                      * end_dist[batch_index][end_pos[batch_index]]
                current_product = start_dist[batch_index][max_start_dist_index[end_dist_index]] \
                                  * end_dist[batch_index][end_dist_index]
                if current_max_product < current_product:
                    start_pos[batch_index] = max_start_dist_index[end_dist_index]
                    end_pos[batch_index] = end_dist_index

            print batch_index, start_pos[batch_index], end_pos[batch_index]

        # # Take argmax to get start_pos and end_post, both shape (batch_size)
        # start_pos = np.argmax(start_dist, axis=1)
        # end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos