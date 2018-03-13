import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_model import QAModel
from modules import *
from rnet_ptrnet_modules import *
from tensorflow.python.ops import rnn_cell

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






        #
        #
        # # Apply fully connected layer to each blended representation
        # # Note, blended_reps_final corresponds to b' in the handout
        # # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        # blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)
        #
        # # Use softmax layer to compute probability distribution for start location
        # # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        # with vs.variable_scope("StartDist"):
        #     softmax_layer_start = SimpleSoftmaxLayer()
        #     self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)
        #
        # # Use softmax layer to compute probability distribution for end location
        # # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        # with vs.variable_scope("EndDist"):
        #     softmax_layer_end = SimpleSoftmaxLayer()
        #     self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


