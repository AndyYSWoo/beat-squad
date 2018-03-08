import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from modules import masked_softmax

class DeepGRU(object):
    def __init__(self, num_layers, hidden_size, keep_prob):
        """
        Inputs:
          num_layers: int. Number of hidden layers
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fws = [DropoutWrapper(rnn_cell.GRUCell(self.hidden_size), input_keep_prob=self.keep_prob)
                             for _ in range(self.num_layers)]
        self.rnn_cell_bws = [DropoutWrapper(rnn_cell.GRUCell(self.hidden_size), input_keep_prob=self.keep_prob)
                             for _ in range(self.num_layers)]

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("DeepBiGRU"):
            # out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size * 2),
            # depth-concatenated forward and backward outputs
            out, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cell_fws, self.rnn_cell_bws, inputs, dtype=tf.float32)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class GatedDotAttn(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("GatedDotAttn"):
            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Blend
            output = tf.concat([keys, output], axis=2)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

        # # Compute gate
        with tf.variable_scope('c2qgate'):
            shape = tf.shape(output)
            dim = output.get_shape().as_list()[-1]
            flatten = tf.reshape(output, (-1, dim))
            W = tf.get_variable('Wc2gate', (dim, dim))
            gate = tf.matmul(flatten, W)
            gate = tf.reshape(gate, shape)
            gate = tf.nn.sigmoid(gate)
            output = gate * output
            return attn_dist, output

