import numpy as np
import tensorflow as tf
import helpers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

tf.reset_default_graph()
sess = tf.InteractiveSession()


vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2



sess.run(tf.global_variables_initializer())
batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
								   vocab_lower=2, vocab_upper=10,
								   batch_size=batch_size)

def next_feed():
	batch = next(batches)
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
	decoder_targets_, _ = helpers.batch(
		[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
	)
	return {
		encoder_inputs: encoder_inputs_,
		encoder_inputs_length: encoder_input_lengths_,
		decoder_targets: decoder_targets_,
	}	