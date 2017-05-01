import numpy as np
import helpers
import sys
import tensorflow as tf
import word_processing as wp
import myconf as cf
from sklearn.cluster import KMeans
import helpers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
'''
def clustering():
	print("start Clustering!!")
	state_vectors = np.loadtxt('./state_vector', dtype=np.float32)
	X = np.array([state_vectors[i] for i in range(186)])
	kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
	labeles = kmeans.labels_
	
	return labels
'''

def learning():
	PAD = 0
	EOS = 1

	vocab_size = 10
	input_embedding_size = 10

	encoder_hidden_units = 10
	decoder_hidden_units = encoder_hidden_units

	encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
	encoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

	embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
	
	encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
	
	
	encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

	encoder_outputs, _ = tf.nn.dynamic_rnn(
		encoder_cell, encoder_inputs_embedded,
		dtype=tf.float32, time_major=True,
	)
	
	logits = tf.contrib.layers.linear(encoder_outputs, vocab_size)
	prediction = tf.argmax(logits, 2)
	
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(encoder_targets, depth=vocab_size, dtype=tf.float32),
			logits=logits,
	)

	loss = tf.reduce_mean(stepwise_cross_entropy)
	train_op = tf.train.AdamOptimizer().minimize(loss)	
	
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())
		##################################################################################################################
		l =  [9, 3, 1, 6, 1, 6, 2, 2, 7, 1, 0, 7, 7, 3, 3, 3, 9, 0, 0, 1, 6, 4, 9, 9, 9, 2, 5, 0, 8, 8, 6, 5, 9, 3, 2, 6, 4, 3, 1, 8, 2, 4, 1, 4, 4, 2, 9, 4, 9, 1, 1, 3, 6, 7, 5, 1, 9, 7, 3, 1, 4, 6, 9, 1, 3, 2, 2, 2, 4, 3, 5, 2, 6, 4, 7, 1, 7, 1, 2, 6, 5, 3, 3, 1, 3, 4, 5, 1, 3, 7, 5, 1, 7, 6, 0, 7, 4, 3, 5, 7, 3, 5, 6, 3, 0, 5, 8, 3, 6, 7, 3, 6, 8, 2, 7, 9, 4, 6, 4, 0, 1, 9, 1, 9, 4, 6, 4, 3, 7, 3, 3, 5, 6, 5, 2, 9, 0, 5, 7, 5, 5, 5, 5, 5, 6, 5, 9, 9, 6, 6, 7, 2, 7, 1, 0, 6, 3, 0, 3, 7, 6, 6, 3, 1, 7, 0, 7, 1, 6, 9, 6, 3, 2, 0, 2, 9, 7, 3, 7, 9, 6, 6, 6, 4, 1, 1]
		batch_ = list()
		
		#1 - 16, 17 - 25, ,26 - 78,79 -134,135 - 169,170 -186	
		print(l[0:16])
		print(l[16:25])
		print(l[25:78])
		print(l[78:134])
		print(l[134:169])
		print(l[169:186])
		
		batch_.append(l[0:16])
		batch_.append(l[16:25])
		batch_.append(l[25:78])
		batch_.append(l[78:134])
		batch_.append(l[134:169])
		batch_.append(l[169:186])
		
		
		encoder_inputs_, _ = helpers.batch(batch_)
		print("-------------------------------------------")
		print_input, _ = helpers.batch(encoder_inputs_)

		
		encoder_targets_, _ = helpers.batch(
					[(sequence[1:]) + [EOS] for sequence in batch_]
				)			
		print_output, _ = helpers.batch(encoder_targets_)
		
		print(print_input)
		print("=============================================")
		print(print_output)
		
		def next_feed():
			encoder_inputs_, _ = helpers.batch(batch_)
			encoder_targets_, _ = helpers.batch(
						[(sequence[1:]) + [EOS] for sequence in batch_]
					)	

			#print("inputs")
			#print(encoder_inputs_)
			
			#print("target..")
			#print(encoder_targets_)
					
			return  {
				encoder_inputs: encoder_inputs_,
				encoder_targets: encoder_targets_,
			}


		max_batches = 4000
		batches_in_epoch = 100
		try:
			for batch in range(max_batches):
				fd = next_feed()
				_, l = sess.run([train_op, loss], fd)

				if batch == 0 or batch % batches_in_epoch == 0:
					print('batch {}'.format(batch))
					print('  minibatch loss: {}'.format(sess.run(loss, fd)))
					predict_ = sess.run(prediction, fd)
					for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if i >= 6:
							break
					print()
		except KeyboardInterrupt:
			print('training interrupted')	
		
if __name__ == '__main__':
	learning()