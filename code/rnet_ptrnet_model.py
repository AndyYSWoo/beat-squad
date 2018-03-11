import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_model import QAModel
from modules import *
from rnet_ptrnet_modules import *
from tensorflow.python.ops import rnn_cell

class RNetPtrModel(QAModel):
    def build_graph(self):
        # Word embedding only TODO char embedding & go deeper
        with tf.variable_scope('Encoding'):
            # 1 layer encoder
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask)  # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask)  # (batch_size, question_len, hidden_size*2)
            # deep encoder
            # encoder = DeepGRU(3, self.FLAGS.hidden_size, self.keep_prob)
            # context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            # question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

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
        blended_reps_final = tf.contrib.layers.fully_connected(self_attention, num_outputs=self.FLAGS.hidden_size) # (batch_size, context_len, hidden_size*2)
        # with tf.variable_scope('PointerNetwork'):
        #     with tf.variable_scope('PointerNetworkInitial'):
        #         V_Q = tf.get_variable('V_Q', shape=(1, context_hiddens.get_shape().as_list()[2]), # (1, hidden_size*2)
        #                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        #         initial_state_layer = MultiplicativeAttnWithParameterKey(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        #         _, initial_state = initial_state_layer.build_graph(question_hiddens, self.qn_mask, V_Q) # (batch_size, 1, hidden_size*2)
        #         # initial_state = tf.squeeze(initial_state, [1]) # (batch_size, hidden_size*2)
        #
        #     with tf.variable_scope('PointerNetworkLayerStart'):
        #         start_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        #         # (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
        #         start_layer_dist, start_layer_output = start_layer.build_graph(self_attention, self.context_mask, initial_state)
        #         with vs.variable_scope("StartDist"):
        #             start_layer_dist = tf.squeeze(start_layer_dist, [1])  # (batch_size, context_len)
        #             self.logits_start = start_layer_dist
        #             self.probdist_start = start_layer_dist
        #
        #     cell = rnn_cell.GRUCell(self.FLAGS.hidden_size*2)
        #     _, new_state = cell(tf.squeeze(start_layer_output, [1]), tf.squeeze(initial_state, [1])) # (batch_size, hidden_size*2)
        #     new_state = tf.expand_dims(new_state, 1) # (batch_size, 1, hidden_size*2)
        #
        #     with tf.variable_scope('PointerNetworkLayerEnd'):
        #         end_layer = MultiplicativeAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        #         # (batch_size, 1, context_len), (batch_size, 1, hidden_size*2)
        #         end_layer_dist, end_layer_output = end_layer.build_graph(self_attention, self.context_mask, new_state)
        #         with vs.variable_scope("EndDist"):
        #             end_layer_dist = tf.squeeze(end_layer_dist, [1])  # (batch_size, context_len)
        #             self.logits_end = end_layer_dist
        #             self.probdist_end = end_layer_dist


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
