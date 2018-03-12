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


class MultiplicativeAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

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
          keys: Tensor shape (batch_size, num_keys, key_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("MultiplicativeAttn"):
            keys_shape = keys.get_shape().as_list() # (batch_size, num_keys, key_vec_size)
            values_shape = values.get_shape().as_list() # (batch_size, num_values, value_vec_size)

            # Calculate attention distribution
            W = tf.get_variable('W_mul_attn', shape=(self.key_vec_size, self.value_vec_size),
                                initializer=tf.contrib.layers.xavier_initializer())
            keys_r = tf.reshape(keys, [-1, keys_shape[2]]) # (batch_size * num_keys, key_vec_size)
            attn_logits = tf.matmul(keys_r, W) # (batch_size * num_keys, value_vec_size)
            attn_logits = tf.reshape(attn_logits, [-1, keys_shape[1], values_shape[2]]) # (batch_size, num_keys, value_vec_size)
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(attn_logits, values_t) # (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            attn_masked_logits, attn_prob_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_prob_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_masked_logits, attn_prob_dist, output


class MultiplicativeAttnWithParameterKey(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

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
          keys: Tensor shape (num_keys, key_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("MultiplicativeAttn"):
            keys_shape = keys.get_shape().as_list() # (num_keys, key_vec_size)
            values_shape = values.get_shape().as_list() # (batch_size, num_values, value_vec_size)

            # Calculate attention distribution
            W = tf.get_variable('W_mul_attn', shape=(self.key_vec_size, self.value_vec_size),
                                initializer=tf.contrib.layers.xavier_initializer())
            values_r = tf.reshape(values, [-1, values_shape[2]]) # shape (batch_size * num_values, value_vec_size)
            attn_logits = tf.matmul(keys, W) # shape (num_keys, value_vec_size)
            attn_logits = tf.matmul(values_r, tf.transpose(attn_logits)) # shape (batch_size * num_values, num_keys)
            attn_logits = tf.reshape(attn_logits, [-1, values_shape[1], keys_shape[0]]) # shape (batch_size, num_values, num_keys)
            attn_logits = tf.transpose(attn_logits, [0, 2, 1]) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


# class AdditiveAttention(object):
#     """Module for basic attention.
#
#     Note: in this module we use the terminology of "keys" and "values" (see lectures).
#     In the terminology of "X attends to Y", "keys attend to values".
#
#     In the baseline model, the keys are the context hidden states
#     and the values are the question hidden states.
#
#     We choose to use general terminology of keys and values in this module
#     (rather than context and question) to avoid confusion if you reuse this
#     module with other inputs.
#     """
#
#     def __init__(self, keep_prob, key_vec_size, value_vec_size, inner_size):
#         """
#         Inputs:
#           keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
#           key_vec_size: size of the key vectors. int
#           value_vec_size: size of the value vectors. int
#           inner_size: d_3 in lecture 11 P.19
#         """
#         self.keep_prob = keep_prob
#         self.key_vec_size = key_vec_size
#         self.value_vec_size = value_vec_size
#         self.inner_size = inner_size
#
#     def build_graph(self, values, values_mask, keys):
#         """
#         Keys attend to values.
#         For each key, return an attention distribution and an attention output vector.
#
#         Inputs:
#           values: Tensor shape (batch_size, num_values, value_vec_size).
#           values_mask: Tensor shape (batch_size, num_values).
#             1s where there's real input, 0s where there's padding
#           keys: Tensor shape (batch_size, num_keys, value_vec_size)
#
#         Outputs:
#           attn_dist: Tensor shape (batch_size, num_keys, num_values).
#             For each key, the distribution should sum to 1,
#             and should be 0 in the value locations that correspond to padding.
#           output: Tensor shape (batch_size, num_keys, hidden_size).
#             This is the attention output; the weighted sum of the values
#             (using the attention distribution as weights).
#         """
#         with vs.variable_scope("AdditiveAttn"):
#             W_1 = tf.get_variable('W_1_{}'.format(values.name), shape=(self.value_vec_size, self.inner_size),
#                                   initializer=tf.contrib.layers.xavier_initializer())
#             W_2 = tf.get_variable('W_2_{}'.format(keys.name), shape=(self.key_vec_size, self.inner_size),
#                                   initializer=tf.contrib.layers.xavier_initializer())
#             v = tf.get_variable('v_{}_{}'.format(values.name, keys.name), shape=(self.inner_size, ),
#                                 initializer=tf.contrib.layers.xavier_initializer())
#             weights = [W_1, W_2]
#             states = [values, keys]
#             outputs = []
#             for weight, state in zip(weights, states):
#                 shape = state.get_shape().as_list()
#                 output = tf.matmul(state, weight)
#                 if len(shape) > 2:
#                     output = tf.reshape(output, (shape[0], shape[1], -1))
#                 elif len(shape) == 2 and shape[0] is None: # TODO
#                     output = tf.reshape(output, (shape[0], 1, -1))
#                 else:
#                     output = tf.reshape(output, (1, shape[0], -1))
#                 outputs.append(output)
#             outputs = sum(outputs)
#             attn_logits = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
#             attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
#             _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values
#
#             # Use attention distribution to take weighted sum of values
#             output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)
#
#             # Apply dropout
#             output = tf.nn.dropout(output, self.keep_prob)
#
#             return attn_dist, output
