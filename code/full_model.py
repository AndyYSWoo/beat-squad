import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from char_model import CharModel
from modules import *
from rnet_ptrnet_modules import *
class FullModel(CharModel):
    def build_graph(self):
        # Encoding input
        with tf.variable_scope('Encoding'):
            # encode chars
            context_input_lens = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.context_char_ids, tf.bool), tf.int32), axis=2), [-1])
            qn_input_lens = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qn_char_ids, tf.bool), tf.int32), axis=2), [-1])
            cell_fw = rnn_cell.GRUCell(self.FLAGS.char_hidden_size)
            cell_bw = rnn_cell.GRUCell(self.FLAGS.char_hidden_size)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.context_char_embs, context_input_lens, dtype=tf.float32)
            ch_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [-1, self.FLAGS.context_len, 2 * self.FLAGS.char_hidden_size])
            self.context_embs = tf.concat([self.context_embs, ch_emb], axis=2)

            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.qn_char_embs, qn_input_lens, dtype=tf.float32)
            qh_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [-1, self.FLAGS.question_len, 2 * self.FLAGS.char_hidden_size])
            self.qn_embs = tf.concat([self.qn_embs, qh_emb], axis=2)
            # deep encoder
            encoder = DeepGRU(3, self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Gated C2Q attention
        with tf.variable_scope('C2QAttention'):
            c2q_attn_layer = GatedDotAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, c2q_attention = c2q_attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # (batch_size, context_len, hidden_size*4)
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            c2q_attention = encoder.build_graph(c2q_attention, self.context_mask) # (batch_size, context_len, hidden_size*2)

        # Self attention
        with tf.variable_scope('SelfMatching'):
            self_attn_layer = GatedDotAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, self_attention = self_attn_layer.build_graph(c2q_attention, self.context_mask, c2q_attention) # (batch_size, context_len, hidden_size*4)
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            self_attention = encoder.build_graph(self_attention, self.context_mask) # (batch_size, context_len, hidden_size*2)

        # Output layer with pointer networks
        with tf.variable_scope('PointerNetwork'):
            with tf.variable_scope('PointerNetworkInitial'):
                V_Q = tf.get_variable('V_Q', shape=(1, context_hiddens.get_shape().as_list()[2]),
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32) # (1, hidden_size*2)
                initial_state_layer = MultiplicativeAttnWithParameterKey(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
                _, initial_state = initial_state_layer.build_graph(question_hiddens, self.qn_mask, V_Q) # (batch_size, 1, hidden_size*2)

            with tf.variable_scope('PointerNetworkLayerStart'):
                start_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
                # (batch_size, 1, context_len), (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
                start_layer_masked_logits, start_layer_prob_dist, start_layer_output = start_layer.build_graph(self_attention, self.context_mask, initial_state)
                with vs.variable_scope("StartDist"):
                    start_layer_masked_logits = tf.squeeze(start_layer_masked_logits, [1])  # (batch_size, context_len)
                    start_layer_prob_dist = tf.squeeze(start_layer_prob_dist, [1])  # (batch_size, context_len)
                    self.logits_start = start_layer_masked_logits
                    self.probdist_start = start_layer_prob_dist

            cell = rnn_cell.GRUCell(self.FLAGS.hidden_size*2)
            _, new_state = cell(tf.squeeze(start_layer_output, [1]), tf.squeeze(initial_state, [1])) # (batch_size, hidden_size*2)
            new_state = tf.expand_dims(new_state, 1) # (batch_size, 1, hidden_size*2)

            with tf.variable_scope('PointerNetworkLayerEnd'):
                end_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
                # (batch_size, 1, context_len), (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
                end_layer_masked_logits, end_layer_prob_dist, end_layer_output = end_layer.build_graph(self_attention, self.context_mask, new_state)
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
        shape = start_dist.shape  # (batch_size, context_len)
        start_pos = np.zeros(shape[0], dtype=np.int64)
        end_pos = np.zeros(shape[0], dtype=np.int64)

        # # Take argmax to get start_pos and end_post, both shape (batch_size)
        # start_pos_old = np.argmax(start_dist, axis=1)
        # end_pos_old = np.argmax(end_dist, axis=1)

        for batch_index in range(shape[0]):
            max_start_dist_index = np.zeros(shape[1], dtype=np.int64)
            max_start_dist_index[0] = 0
            for start_dist_index in range(1, shape[1]):
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

                    # print batch_index, start_pos[batch_index], end_pos[batch_index], \
                    #     start_dist[batch_index][start_pos[batch_index]] * end_dist[batch_index][end_pos[batch_index]], \
                    #     start_pos_old[batch_index], end_pos_old[batch_index], \
                    #     start_dist[batch_index][start_pos_old[batch_index]] * end_dist[batch_index][end_pos_old[batch_index]]

        return start_pos, end_pos
