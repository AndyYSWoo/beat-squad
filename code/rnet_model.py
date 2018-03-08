import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_model import QAModel
from modules import *
from rnet_modules import *

class RNetModel(QAModel):
    def build_graph(self):
        # Word embedding only TODO char embedding & go deeper
        with tf.variable_scope('Encoding'):
            encoder = DeepGRU(3, self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Gated C2Q attention
        with tf.variable_scope('C2QAttention'):
            c2q_attn_layer = GatedDotAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, c2q_attention = c2q_attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # (batch_size, context_len, hidden_size*4)
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            c2q_attention = encoder.build_graph(c2q_attention, self.context_mask) # (batch_size, context_len, hidden_size*2)

        with tf.variable_scope('SelfMatching'):
            self_attn_layer = GatedDotAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, self_attention = self_attn_layer.build_graph(c2q_attention, self.context_mask, c2q_attention) # (batch_size, context_len, hidden_size*4)
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            self_attention = encoder.build_graph(self_attention, self.context_mask) # (batch_size, context_len, hidden_size*2)

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(self_attention, num_outputs=self.FLAGS.hidden_size) # (batch_size, context_len, hidden_size*2)

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
