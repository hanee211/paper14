import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf


def test():
	print("start Testing!!")
	
	restore = True

	args = sys.argv
	args = args[1:]
	
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-r':
			restore = value


	PAD = 0 
	EOS = 1
	

	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	encoder_hidden_units = cf.encoder_hidden_units
	#batch_size = cf.batch_size
	#############################################################################
	batch_size = 2
	#############################################################################
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = encoder_hidden_units
	params['batch_size'] = batch_size	
	model = Model(params, training = False)
	saver = tf.train.Saver()
	
	############################################################################	
	idx2word, word2idx = wp.get_wordListFromFile()
	sentences = wp.get_sentences()
	
	max_encoder_length = 0
	for b in sentences:
		if len(b) > max_encoder_length:
			max_encoder_length = len(b)			
	#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
	max_decoder_length = max_encoder_length + 3	
	
	'''
	encoder_input_list = list()
	encoder_input_length_list = list()
	decoder_target_list = list()
	decoder_length_list = list()


	for i in range(int(len(sentences)/batch_size)):
		start = i * batch_size
		end = start + batch_size

		batch = sentences[start:end]

		
		max_encoder_length = 0
		for b in batch:
			if len(b) > max_encoder_length:
				max_encoder_length = len(b)			
		#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
		max_decoder_length = max_encoder_length + 3
		
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		decoder_targets_, _ = helpers.batch(
			#[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
			[(sequence) + [EOS] + [PAD] *(max_decoder_length - len(sequence) -1 ) for sequence in batch]
		)

		encoder_input_list.append(encoder_inputs_)
		encoder_input_length_list.append(encoder_input_lengths_)
		decoder_target_list.append(decoder_targets_)
		decoder_length_list.append([max_decoder_length for v in encoder_input_lengths_])	
	'''
	
	
	##########################################################################
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		#with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

		model_ckpt_file = './status/model.ckpt'
		saver.restore(sess, model_ckpt_file)
		print("Setting done.")		

		#########################################################################
		'''
		batch = wp.get_sentences()
		idx2word, _ = wp.get_wordList()
		max_encoder_length = 0
		for b in batch:
			if len(b) > max_encoder_length:
				max_encoder_length = len(b)			
		#batch = [[3,4,5,2,6,7,8,9],[6,7,3,4,5],[2,2,4,5,6]]
		max_decoder_length = max_encoder_length + 3		
		
		#########################################################################
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		'''

		'''
		with open('state_vector', 'w') as f:
			for i in range(int(len(sentences)/batch_size)):		
			#for i in range(1):		
				fd = {model.encoder_inputs: encoder_input_list[i],
					model.encoder_inputs_length: encoder_input_length_list[i],}	
				_state = sess.run(model.encoder_final_state, fd)
				#print(_state)
				for j in range(len(_state[0])):
					f.write(' '.join(str(vv) for vv in _state[0][j]))
					f.write(' ')
					f.write(' '.join(str(vv) for vv in _state[1][j]))
					f.write('\n')
				
			
		'''
		
		state_vectors = np.loadtxt('./state_vector', dtype=np.float32)
		i = 0
		
		c_list = list()
		h_list = list()
		
		index = 10
		for i in range(2):
			i = i + index
			_c = list(state_vectors[i][0:256])
			_h = list(state_vectors[i][256:512])
			
			c_list.append(_c)
			h_list.append(_h)
		
		
		fd_predict = {model.state_c : c_list, model.state_h : h_list
						,model.decoder_lengths :  [max_decoder_length for v in range(batch_size)]
					}
		
		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		#print(sess.run(model.direct_encoder_final_state, fd_predict))
		predict_ = sess.run(model.decoder_prediction, fd_predict)
		for l in predict_.T:
			print(l)
			_s = list()
			for w in l:
				if idx2word[w] != 'EOS' and idx2word[w] != 'PAD':
					_s.append(idx2word[w])
			print(" ".join(_s))
		print("==========================")
		

if __name__ == '__main__':
	test()		