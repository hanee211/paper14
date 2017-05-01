import helpers

batch_size = 3

EOS = 1
PAD = 0

batches = helpers.random_sequences(length_from=3, length_to=8,
								   vocab_lower=2, vocab_upper=10,
								   batch_size=batch_size)
								   
def next_feed():
	print("next feed")
	print("--------------------------------")
	batch = next(batches)
	print(batch)
	
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
	decoder_targets_, _ = helpers.batch(
		[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
	)
	return {
		'encoder_inputs': encoder_inputs_,
		'encoder_inputs_length': encoder_input_lengths_,
		'decoder_targets': decoder_targets_,
	}	
	
loss_track = []

max_batches = 2
batches_in_epoch = 1000

for batch in range(max_batches):
	fd = next_feed()
	print("#####################")
	print(fd)