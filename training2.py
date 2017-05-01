import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import datetime as dt

def train():
	print("start training!!")
	
	restore = True

	args = sys.argv
	args = args[1:]
	
	PAD = 0 
	EOS = 1
		
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-r':
			restore = value

	print(restore)	
	
	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	encoder_hidden_units = cf.encoder_hidden_units
	batch_size = cf.batch_size
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = encoder_hidden_units
	params['batch_size'] = batch_size
	
	model = Model(params)
	saver = tf.train.Saver()

	sentences = wp.get_sentences()

	max_encoder_length = 0
	for b in sentences:
		if len(b) > max_encoder_length:
			max_encoder_length = len(b)			
	#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
	max_decoder_length = max_encoder_length + 3	
	
	encoder_input_list = list()
	encoder_input_length_list = list()
	decoder_target_list = list()
	decoder_length_list = list()
	

	for i in range(int(len(sentences)/batch_size)):
		start = i * batch_size
		end = start + batch_size

		batch = sentences[start:end]

		'''
		max_encoder_length = 0
		for b in batch:
			if len(b) > max_encoder_length:
				max_encoder_length = len(b)			
		#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
		max_decoder_length = max_encoder_length + 3
		'''
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		decoder_targets_, _ = helpers.batch(
			#[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
			[(sequence) + [EOS] + [PAD] *(max_decoder_length - len(sequence) -1 ) for sequence in batch]
		)

		encoder_input_list.append(encoder_inputs_)
		encoder_input_length_list.append(encoder_input_lengths_)
		decoder_target_list.append(decoder_targets_)
		decoder_length_list.append([max_decoder_length for v in encoder_input_lengths_])

	
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		#with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

		model_ckpt_file = './status/model.ckpt'

		print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
		print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
		print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
		if restore == 'T': 
			print("restoring.... ")
			saver.restore(sess, model_ckpt_file)		
		else:
			print("not restoring....")

		batches = helpers.random_sequences(length_from=3, length_to=8,
										   vocab_lower=2, vocab_upper=10,
										   batch_size=batch_size)
										   
										   
		def next_feed():
			batch = next(batches)
			#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
			max_decoder_length = max_encoder_length + 3
			
			encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
			decoder_targets_, _ = helpers.batch(
				#[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
				[(sequence) + [EOS] + [PAD] *(max_decoder_length - len(sequence) -1 ) for sequence in batch]
			)

			return {
				model.encoder_inputs: encoder_inputs_,
				model.encoder_inputs_length: encoder_input_lengths_,
				model.decoder_targets: decoder_targets_,
				model.decoder_lengths : [max_decoder_length for v in encoder_input_lengths_]
			}
			
		def next_feed_word(start, end):
			batch = sentences[start:end]
			'''
			max_encoder_length = 0
			for b in batch:
				if len(b) > max_encoder_length:
					max_encoder_length = len(b)			
			#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
			max_decoder_length = max_encoder_length + 3
			'''
			encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
			decoder_targets_, _ = helpers.batch(
				#[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
				[(sequence) + [EOS] + [PAD] *(max_decoder_length - len(sequence) -1 ) for sequence in batch]
			)

			return {
				model.encoder_inputs: encoder_inputs_,
				model.encoder_inputs_length: encoder_input_lengths_,
				model.decoder_targets: decoder_targets_,
				model.decoder_lengths : [max_decoder_length for v in encoder_input_lengths_]
			}				
			
		loss_track = []

		max_batches = 51
		batches_in_epoch = 3

		try:
			for e in range(max_batches):
				start_time_out = dt.datetime.now()
				#print(batch)
				print(e, " epoch start...")
				#fd = next_feed()
				for i in range(int(len(sentences)/batch_size)):
					start_time = dt.datetime.now()
					
					#start = i * batch_size
					#end = start + batch_size
					#print("get data")
					#fd = next_feed_word(start,end)
	
					fd = {
						model.encoder_inputs: encoder_input_list[i],
						model.encoder_inputs_length: encoder_input_length_list[i],
						model.decoder_targets: decoder_target_list[i],
						model.decoder_lengths : decoder_length_list[i]
					}						
					
					print("batch processing...")
					_, l = sess.run([model.train_op, model.loss], fd)
					#_, l = sess.run([train_op_gd, loss], fd)
					print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", str(i) , " in ", str(len(sentences)/batch_size))
				
				print("Take", str((dt.datetime.now() - start_time_out).seconds), "seconds for in epoch. current is ", str(e))
				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
					predict_ = sess.run(model.decoder_prediction, fd)
					for i, (inp, pred) in enumerate(zip(fd[model.encoder_inputs].T, predict_.T)):
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if i >= 10:
							break
			
					saver.save(sess, model_ckpt_file)
					print("mode saved to ", model_ckpt_file)

		except KeyboardInterrupt:
			print('training interrupted')

if __name__ == '__main__':
	train()		